import os
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup

CHROMA_DIR = "./chroma_db"
SOURCE_URL = "https://www.primarycarepages.sg/healthier-sg/care-protocols/chronic-care-protocols/"

PROMPT_TEMPLATE = """Answer the question based only on the following context.
If you don't know, say you don't know.

{context}

Question: {question}
"""


def _build_vectorstore() -> Chroma:
    loader = RecursiveUrlLoader(
        url=SOURCE_URL,
        max_depth=2,
        extractor=lambda x: BeautifulSoup(x, "html.parser").text,
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def ingest() -> int:
    """Scrape the source, embed, and persist. Returns number of chunks."""
    loader = RecursiveUrlLoader(
        url=SOURCE_URL,
        max_depth=2,
        extractor=lambda x: BeautifulSoup(x, "html.parser").text,
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return len(splits)


def build_chain(vectorstore: Chroma):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    answer_chain = prompt | llm | StrOutputParser()
    return RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough(),
    }).assign(answer=answer_chain)
