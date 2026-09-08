import os
import hashlib
import io
import requests
from urllib.parse import urlsplit
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableLambda, RunnableParallel, RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from context.prompts import format_live_context

CHROMA_DIR = "./chroma_db"

MAX_PDF_BYTES = 25 * 1024 * 1024  # guideline PDFs run 1-10 MB; bounded upload cap


def pdf_text(data: bytes) -> str:
    """Extract readable text from a PDF, one labelled block per page.

    Raises ValueError for password-protected PDFs. Returns "" when the
    PDF has no text layer (scanned image) — caller treats that as a
    reject.
    """
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        if not reader.decrypt(""):
            raise ValueError("PDF is password-protected")
    blocks = []
    for i, page in enumerate(reader.pages, start=1):
        t = (page.extract_text() or "").strip()
        if t:
            blocks.append("[page %d]\n%s" % (i, t))
    return "\n\n".join(blocks)

# LLM provider — OpenAI-compatible. Defaults to api.openai.com; point
# OPENAI_BASE_URL at a gateway (e.g. https://openrouter.ai/api/v1) to use one.
# Model ids must match the provider's namespace (OpenRouter: "openai/gpt-4o").
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")

# --- corpus sources (Phase 2: multi-source) ---------------------------------------
# (name, url, max_depth). `name` becomes the chunk metadata `source_site`.
#   primarycarepages.sg — the 17 chronic-care protocols (depth-2 crawl, as before).
#   moh.gov.sg — MOH haze public guidance, ONE static page at depth 0 (moh.gov.sg
#     is huge — never crawl it). Citable, but NEVER as protocol content: it is
#     rendered in its own "Public health guidance" prompt section.
SOURCES = [
    ("primarycarepages.sg",
     "https://www.primarycarepages.sg/healthier-sg/care-protocols/chronic-care-protocols/", 2),
    ("moh.gov.sg", "https://www.moh.gov.sg/others/haze/", 0),
]
# Chunk sites rendered in the "Protocol content" prompt section: everything NOT
# listed here (and chunks with no source_site, i.e. the pre-Phase 2 corpus) is
# public guidance — never presented as protocol content.
PUBLIC_GUIDANCE_SITES = {"moh.gov.sg"}

# gov.sg sits behind CloudFront and 403s the default python-requests UA.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Isomer (gov.sg) boilerplate headings — drop the whole section, not just the heading.
_ISOMER_BOILERPLATE_HEADINGS = ("other pages in", "related sites", "back to top",
                                "useful links")


def _plain_text(html: str) -> str:
    """Whole-page text — primarycarepages behaviour, unchanged."""
    return BeautifulSoup(html, "html.parser").text


def _isomer_text(html: str) -> str:
    """Content-node-targeted text for gov.sg (Isomer) pages.

    Targets <main id="main-content"> (fallback: <main> / [role=main] / body)
    and strips Isomer boilerplate:
      - script/style/nav/header/footer elements
      - link-list sections under "Other pages in …" / "Related sites" /
        "Back to top" / "Useful links" headings (h1-h4, and <summary>
        accordions -> the whole <details>)
      - the site-footer column that holds the "Back to top" button
        ("Other pages in …" / "See all pages" link lists)
      - "(opens in new tab)" markers
    Verified against the MOH haze page (moh.gov.sg/others/haze/).
    """
    soup = BeautifulSoup(html, "html.parser")
    node = (soup.find("main", id="main-content") or soup.find("main")
            or soup.find(attrs={"role": "main"}) or soup.body or soup)
    for tag in node.find_all(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    for h in node.find_all(["h1", "h2", "h3", "h4", "summary"]):
        text = h.get_text(" ", strip=True).lower()
        if any(text.startswith(p) for p in _ISOMER_BOILERPLATE_HEADINGS):
            target = h.find_parent("details") or h.find_parent("section") or h.parent or h
            target.decompose()
    for b in node.find_all("button"):
        if b.get_text(" ", strip=True).lower() == "back to top":
            col = b
            while col is not None and col is not node:
                classes = col.get("class") or []
                if any(str(c).startswith("col-span-") for c in classes):
                    break
                col = col.parent
            (col or b).decompose()
    for el in node.find_all(string=lambda s: s and s.strip() == "(opens in new tab)"):
        el.extract()
    return node.get_text("\n", strip=True)


def _fetch(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=60)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def _load_source(name: str, url: str, max_depth: int) -> list:
    """Load one source into Documents carrying `source_site` metadata."""
    if max_depth == 0:
        return [Document(page_content=_isomer_text(_fetch(url)),
                         metadata={"source": url, "source_site": name})]
    loader = RecursiveUrlLoader(url=url, max_depth=max_depth, extractor=_plain_text)
    docs = loader.load()
    for d in docs:
        d.metadata["source_site"] = name
    return docs


def _split(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def _load_all_sources() -> list:
    docs = []
    for name, url, depth in SOURCES:
        docs.extend(_load_source(name, url, depth))
    return docs


def _norm_url(u: str) -> str:
    """Normalize for dedupe: lowercase scheme/host/path, drop trailing slash,
    ignore query/fragment."""
    if not u:
        return ""
    p = urlsplit(u.strip())
    return "%s://%s%s" % (p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/").lower())


def _store_url_index(vectorstore: Chroma):
    """(normalized-URL set, host set) of chunks already in the store
    (local read, 0 credits)."""
    urls, hosts = set(), set()
    for m in vectorstore.get(include=["metadatas"]).get("metadatas") or []:
        u = (m or {}).get("source")
        if not u:
            continue
        urls.add(_norm_url(u))
        hosts.add(urlsplit(u).netloc.lower())
    return urls, hosts


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL, openai_api_base=OPENAI_BASE_URL)


PROMPT_TEMPLATE = """Answer the question using only the four sections below. If they do not cover the question, say you don't know.

- "Protocol content": excerpts from the clinic's chronic-care protocols. This is the only material you may cite as protocol content. Where basis notes are given they state exactly what the protocol text does and does not support — do not over-claim beyond them.
- "Clinical guidelines": Singapore ACE clinical guidelines (ACGs) and other uploaded clinical guidance PDFs. Cite them as guideline content — never as protocol content.
- "Public health guidance": government public health guidance (e.g. MOH). You may cite it, but as public guidance — never as protocol content.
- "Local context": live population-level signals (air quality, weather, dengue) for the clinic point. These are observations about the area right now, NOT protocol content: use them to frame the answer (environmental triggers, sick-day rules, counselling), but never attribute them to the protocols.

Formatting: the answer renders as PLAIN TEXT in a chat window — it has NO markdown support. Never use **bold**, *italic*, # headings, or backticks; write plain words only. Simple dash bullets and a blank line between sections are fine.

== Protocol content ==
{protocol_content}

== Clinical guidelines ==
{guidelines}

== Public health guidance ==
{public_guidance}

== Local context — clinic: {clinic}, as of {as_of} ==
{local_context}

Question: {question}
"""


def load_vectorstore() -> Chroma:
    embeddings = _embeddings()
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def ingest() -> int:
    """Full rebuild from ALL sources (costs credits for the whole corpus).

    NOTE: Chroma.from_documents appends into an existing persisted collection,
    so this is meant for a fresh ./chroma_db. To add new pages to the live
    store without re-embedding what is already there, use ingest_append().
    """
    splits = _split(_load_all_sources())
    embeddings = _embeddings()
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return len(splits)


def ingest_append(max_new_chunks: int = 500):
    """Append-only ingest (Phase 2): add sources the store has never seen,
    matched by HOST; within a new source, skip chunks whose URL (normalized)
    is already stored. Returns (new_chunk_count, sources_added, sources_skipped).

    Why host-level skip: the site may rewrite its URL structure (primarycarepages
    now redirects /chronic-care-protocols/ -> /chronic/ with case changes), which
    makes per-URL matching unreliable for sites already in the store — matching
    them by URL would re-embed the whole corpus. A site already covered is
    never touched; use a full ingest() rebuild to refresh an existing site.
    """
    vectorstore = load_vectorstore()
    have_urls, have_hosts = _store_url_index(vectorstore)
    new, added, skipped = [], [], []
    for name, url, depth in SOURCES:
        if urlsplit(url).netloc.lower() in have_hosts:
            skipped.append(name)
            continue
        for d in _split(_load_source(name, url, depth)):
            if _norm_url(d.metadata.get("source")) in have_urls:
                continue
            new.append(d)
        added.append(name)
    if len(new) > max_new_chunks:
        raise RuntimeError(
            "append ingest would embed %d chunks (> %d cap) — refusing to "
            "burn credits; check SOURCES" % (len(new), max_new_chunks))
    if new:
        vectorstore.add_documents(new)
    return len(new), added, skipped


# Phase 5: clinical-guideline PDFs (ACE ACGs + manual uploads) route to the
# "Clinical guidelines" prompt section — never presented as protocol content.
GUIDELINE_SITES = {"ace-hta.gov.sg", "guideline-upload"}


def _store_pdf_index(vectorstore: Chroma):
    """(doc_hash set, normalized source-URL set) already in the store
    (local read, 0 credits) — dedupe for PDF ingest."""
    hashes, urls = set(), set()
    for m in vectorstore.get(include=["metadatas"]).get("metadatas") or []:
        m = m or {}
        if m.get("doc_hash"):
            hashes.add(m["doc_hash"])
        u = m.get("source")
        if u:
            urls.add(_norm_url(u))
    return hashes, urls


def ingest_pdf(data: bytes, title: str, source: str = "",
               source_site: str = "guideline-upload"):
    """Parse + chunk + embed + append one PDF. Dedupes by sha256 of the file
    bytes (and by source URL when given), so re-ingests cost 0 credits.

    Returns (chunks_added, skipped, doc_hash). Raises ValueError when the
    PDF has no extractable text (scanned image / password-protected).
    """
    text = pdf_text(data)
    if not text.strip():
        raise ValueError("no extractable text in PDF (scanned image?)")
    doc_hash = hashlib.sha256(data).hexdigest()[:16]
    vectorstore = load_vectorstore()
    have_hashes, have_urls = _store_pdf_index(vectorstore)
    src = source or "upload:%s" % title
    if doc_hash in have_hashes or _norm_url(src) in have_urls:
        return 0, True, doc_hash
    doc = Document(
        page_content=text,
        metadata={"source": src, "source_site": source_site,
                  "doc_title": title, "doc_hash": doc_hash},
    )
    splits = _split([doc])
    vectorstore.add_documents(splits)
    return len(splits), False, doc_hash


def list_pdfs():
    """One row per ingested PDF: doc_hash, title, source, chunks."""
    vectorstore = load_vectorstore()
    agg = {}
    for m in vectorstore.get(include=["metadatas"]).get("metadatas") or []:
        m = m or {}
        if not m.get("doc_hash"):
            continue
        e = agg.setdefault(m["doc_hash"], {
            "title": m.get("doc_title", "?"),
            "source": m.get("source", ""),
            "chunks": 0})
        e["chunks"] += 1
    return [{"doc_hash": k, "title": v["title"], "source": v["source"],
             "chunks": v["chunks"]} for k, v in agg.items()]


def _active_protocol_names(context_provider) -> list:
    """Mechanism B: protocol names engaged by the active live links."""
    if not context_provider:
        return []
    try:
        snap = context_provider()
    except Exception:
        return []
    names = []
    for link in ((snap or {}).get("protocol_links") or {}).get("active") or []:
        for p in link.get("protocols") or []:
            if p not in names:
                names.append(p)
    return names


def _steer_query(question: str, context_provider) -> str:
    """Append active protocol names for the RETRIEVER only — the LLM prompt
    keeps the original question."""
    names = _active_protocol_names(context_provider)
    if not names:
        return question
    return question + "\n[related chronic-care protocols: %s]" % ", ".join(names)


def _prepare_sections(docs: list, question: str, context_provider) -> dict:
    """Split retrieved chunks by provenance and attach the live-context
    sections (basis rule: `derived`/`none` links never become protocol content)."""
    protocol_parts, public_parts, guideline_parts = [], [], []
    for d in docs or []:
        site = d.metadata.get("source_site", "primarycarepages.sg")
        body = (d.page_content or "").strip()
        if not body:
            continue
        ref = "Source: %s" % d.metadata.get("source", "")
        if site in PUBLIC_GUIDANCE_SITES:
            public_parts.append(
                "%s\n[%s — public health guidance, not clinic protocol content] %s"
                % (body, site, ref))
        elif site in GUIDELINE_SITES:
            guideline_parts.append(
                "%s\n[%s] %s"
                % (body, d.metadata.get("doc_title", "guideline"), ref))
        else:
            protocol_parts.append("%s\n[%s] %s" % (body, site, ref))

    snapshot = None
    if context_provider:
        try:
            snapshot = context_provider()
        except Exception:
            snapshot = None
    if snapshot:
        basis_notes, local_ctx, clinic, as_of = format_live_context(snapshot)
    else:
        basis_notes = None
        local_ctx = ("(live context unavailable — snapshot not ready; answer from "
                     "the protocol and public-guidance sections only)")
        clinic = as_of = "n/a"

    if basis_notes:
        protocol_parts.append(
            "Basis notes for the currently active local signals — what the "
            "protocol text does and does not support:\n" + basis_notes)
    return {
        "protocol_content": ("\n\n---\n\n".join(protocol_parts)
                             or "(no protocol content retrieved for this question)"),
        "guidelines": ("\n\n---\n\n".join(guideline_parts)
                       or "(no clinical guidelines retrieved for this question)"),
        "public_guidance": "\n\n---\n\n".join(public_parts) or "(none)",
        "local_context": local_ctx,
        "clinic": clinic,
        "as_of": as_of,
        "question": question,
    }


def build_chain(vectorstore: Chroma, context_provider=None):
    """context_provider: callable() -> snapshot dict or None. app.py serves it
    from the same 15-min cache as GET /api/context; the chat path never blocks
    on a build (a cold cache degrades the prompt to 'context unavailable')."""
    # Bounded provider calls: the free tier is slow and 429-prone; without
    # explicit caps a hung request would sit for ~30 min (openai client
    # defaults: 600 s/attempt x 3 tries) before surfacing as an error.
    llm = ChatOpenAI(model_name=CHAT_MODEL, openai_api_base=OPENAI_BASE_URL,
                     temperature=0, timeout=300, max_retries=1)
    # k=5 was tuned for the small protocol corpus; with the Phase 5 guideline
    # PDFs (~3k chunks, a single ACG can span 30+ chunks) 5 retrieved chunks
    # surfaced only titles/intro text for longer guidelines. 12 keeps the
    # prompt well under the context window (~15k chars) while covering a
    # guideline's relevant section.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    answer_chain = prompt | llm | StrOutputParser()

    def _run(state: dict) -> dict:
        docs = state.get("context") or []
        question = state["question"]
        answer = answer_chain.invoke(_prepare_sections(docs, question, context_provider))
        return {"answer": answer, "question": question, "context": docs}

    return (
        RunnableParallel({
            "context": (RunnableLambda(lambda q: _steer_query(q, context_provider))
                        | retriever),
            "question": RunnablePassthrough(),
        })
        | RunnableLambda(_run)
    )
