"""
SourceMatch — Upload your outline. Upload your sources. Get real, verified quotes.

A Streamlit app that matches verbatim quotes from uploaded PDFs to sections of
a paper outline, using Gemini for ranking and Python for exact-match verification.
Zero hallucination: every exported quote is substring-verified against the source.
"""

import io
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field, asdict

import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader


# ---------- setup ----------

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Primary model, then fallbacks if the primary is overloaded.
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]


@st.cache_resource
def get_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def gemini_generate(contents: str, *, json_mode: bool = False, max_retries: int = 4):
    """Call Gemini with retries on 503/overload and model fallback."""
    config = {"response_mime_type": "application/json"} if json_mode else None
    last_err: Exception | None = None
    for model in GEMINI_MODELS:
        delay = 2.0
        for attempt in range(max_retries):
            try:
                return get_client().models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                last_err = e
                msg = str(e)
                # Retry on transient overload / rate-limit / server errors.
                transient = any(
                    s in msg
                    for s in ("503", "UNAVAILABLE", "overload", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")
                )
                if not transient:
                    raise
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                    continue
                # Out of retries on this model — break to try the next model.
                break
    raise RuntimeError(
        f"Service unavailable after retries: {last_err}"
    )


# ---------- data model ----------


@dataclass
class Chunk:
    id: int
    doc: str
    page: int
    text: str


@dataclass
class Quote:
    section: str
    text: str
    doc: str
    page: int
    accepted: bool = True  # default accepted; user can reject


# ---------- PDF parsing (deterministic, no LLM) ----------


def parse_pdf(file_bytes: bytes, filename: str, start_id: int) -> list[Chunk]:
    """Extract one chunk per page from a PDF."""
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks: list[Chunk] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 40:
            continue  # skip cover pages / images
        chunks.append(
            Chunk(id=start_id + len(chunks), doc=filename, page=i + 1, text=text)
        )
    return chunks


def parse_url(url: str, start_id: int) -> list[Chunk]:
    """Fetch a URL, strip HTML, split into ~2000-char 'pages'."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8", errors="ignore")
    # Drop scripts/styles before tag-stripping.
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Use the page <title> as the source name when available.
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.DOTALL | re.IGNORECASE)
    if m:
        name = re.sub(r"\s+", " ", m.group(1)).strip()[:80]
    else:
        name = url.split("/")[2] if "//" in url else url
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&\w+;", " ", text)  # strip basic HTML entities
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[Chunk] = []
    size = 2000
    for i in range(0, len(text), size):
        piece = text[i : i + size].strip()
        if len(piece) < 40:
            continue
        chunks.append(
            Chunk(
                id=start_id + len(chunks),
                doc=name,
                page=(i // size) + 1,
                text=piece,
            )
        )
    return chunks


# ---------- Gemini helpers ----------


def _extract_json(text: str):
    """Tolerate markdown code fences around JSON responses."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def generate_outline_from_description(description: str) -> str:
    prompt = f"""You are helping a student plan a research paper.
Given this short description, produce a clear paper outline with a title and
4-5 Roman-numeral sections, each with a one-line theme.

Description:
{description}

Return plain text only, no markdown fences. Example format:

Title Goes Here

I. Introduction — short theme
II. Literature Review — short theme
III. Analysis — short theme
IV. Conclusion — short theme
"""
    resp = gemini_generate(prompt)
    return (resp.text or "").strip()


def analyze_outline(outline: str) -> list[dict]:
    """Return [{'section': str, 'evidence_need': str}, ...]."""
    prompt = f"""Read this paper outline and return a JSON array describing each
section. For each section include:
  - "section": the section heading as it appears in the outline
  - "evidence_need": one sentence describing what kind of evidence/quotes would
    support that section.

Outline:
{outline}

Return ONLY a JSON array. No prose, no code fences.
"""
    resp = gemini_generate(prompt, json_mode=True)
    try:
        data = _extract_json(resp.text or "[]")
        if isinstance(data, list):
            return [
                {"section": d["section"], "evidence_need": d.get("evidence_need", "")}
                for d in data
                if "section" in d
            ]
    except Exception:
        pass
    # Fallback: regex out Roman-numeral lines
    sections = []
    for line in outline.splitlines():
        if re.match(r"^\s*[IVX]+\.\s+\S", line):
            sections.append({"section": line.strip(), "evidence_need": ""})
    return sections


def rank_quotes_for_section(
    section: str, evidence_need: str, chunks: list[Chunk], top_k: int = 5
) -> list[dict]:
    """Ask Gemini for the best verbatim quotes from the indexed chunks."""
    # Pack chunks into a numbered index. Truncate each to keep prompt small.
    lines = []
    for c in chunks:
        snippet = c.text[:1200]
        lines.append(f"[{c.id}] ({c.doc} p.{c.page}) {snippet}")
    index = "\n\n".join(lines)

    prompt = f"""You are SourceMatch. Find the best supporting quotes for this
paper section from the indexed passages below.

SECTION: {section}
EVIDENCE NEED: {evidence_need}

Rules:
1. Return up to {top_k} quotes.
2. Each "quote" MUST be copied VERBATIM from the corresponding passage text.
   Do not paraphrase. Do not fix typos. Do not add ellipses.
3. Prefer specific claims, statistics, and direct statements.
4. Use the passage id exactly as shown in brackets.

Return ONLY a JSON array of objects like:
[{{"id": 3, "quote": "exact verbatim text"}}, ...]

INDEXED PASSAGES:
{index}
"""
    resp = gemini_generate(prompt, json_mode=True)
    try:
        data = _extract_json(resp.text or "[]")
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


# ---------- verification (Python, exact substring) ----------


def verify_quotes(candidates: list[dict], chunks_by_id: dict[int, Chunk]) -> list[Quote]:
    """Drop any quote that is not an exact substring of its cited chunk."""
    verified: list[Quote] = []
    for item in candidates:
        try:
            cid = int(item["id"])
            quote = str(item["quote"]).strip().strip('"""').strip()
        except Exception:
            continue
        chunk = chunks_by_id.get(cid)
        if not chunk or not quote:
            continue
        # Normalize whitespace on both sides for robust matching.
        norm_quote = re.sub(r"\s+", " ", quote)
        norm_text = re.sub(r"\s+", " ", chunk.text)
        if norm_quote and norm_quote in norm_text:
            verified.append(
                Quote(section="", text=norm_quote, doc=chunk.doc, page=chunk.page)
            )
    return verified


# ---------- citation export ----------


def format_citation(q: Quote, style: str) -> str:
    name = q.doc.rsplit(".", 1)[0].replace("_", " ")
    if style == "APA":
        return f'"{q.text}" ({name}, p. {q.page}).'
    if style == "MLA":
        return f'"{q.text}" ({name} {q.page}).'
    if style == "Chicago":
        return f'"{q.text}" ({name}, {q.page}).'
    return f'"{q.text}" — {q.doc}, p. {q.page}'


def build_export(quotes_by_section: dict[str, list[Quote]], style: str) -> str:
    out = [f"SourceMatch Export — {style}", "=" * 40, ""]
    for section, quotes in quotes_by_section.items():
        accepted = [q for q in quotes if q.accepted]
        if not accepted:
            continue
        out.append(section)
        out.append("-" * len(section))
        for q in accepted:
            out.append(format_citation(q, style))
            out.append("")
        out.append("")
    return "\n".join(out)


# ---------- in-document insertion + inline citation rendering ----------


def insert_quote_into_paper(q: Quote) -> None:
    """Drop a verbatim quote + cite marker under its section heading in ss.paper."""
    doc_name = q.doc.rsplit(".", 1)[0].replace("_", " ")
    snippet = f'"{q.text}" [cite:{doc_name}:{q.page}]'
    paper = st.session_state.paper
    idx = paper.find(q.section)
    if idx == -1:
        st.session_state.paper = (
            paper.rstrip() + f"\n\n## {q.section}\n\n{snippet}\n"
        )
    else:
        line_end = paper.find("\n", idx)
        if line_end == -1:
            line_end = len(paper)
        st.session_state.paper = paper[:line_end] + f"\n\n{snippet}" + paper[line_end:]


_CITE_RE = re.compile(r"\[cite:([^:\]]+):(\d+)\]")


def render_paper_with_citations(paper: str) -> str:
    """Replace [cite:doc:page] markers with numbered superscripts and append References."""
    refs: list[tuple[str, int]] = []
    seen: dict[tuple[str, int], int] = {}

    def replace(m):
        key = (m.group(1), int(m.group(2)))
        if key not in seen:
            refs.append(key)
            seen[key] = len(refs)
        return f"<sup>[{seen[key]}]</sup>"

    body = _CITE_RE.sub(replace, paper)
    if not refs:
        return body
    lines = ["", "---", "", "**References**", ""]
    for i, (doc, page) in enumerate(refs, 1):
        lines.append(f"{i}. {doc}, p. {page}")
    return body + "\n" + "\n".join(lines)


def write_paper_from_bucket(outline: str, bucket: list[Quote]) -> str:
    """Ask Gemini to draft the full paper using the outline + selected quotes."""
    if not bucket:
        return outline
    quote_lines = []
    for q in bucket:
        doc_name = q.doc.rsplit(".", 1)[0].replace("_", " ")
        marker = f"[cite:{doc_name}:{q.page}]"
        quote_lines.append(f'- Section "{q.section}": "{q.text}" {marker}')
    quotes_block = "\n".join(quote_lines)
    prompt = f"""You are drafting a research paper. Use the outline below and weave
in the provided verbatim quotes naturally within their assigned sections.

Rules:
1. Use the outline structure. Render section headings as markdown (## Heading).
2. Each provided quote MUST appear VERBATIM in its assigned section, wrapped in
   double quotes, immediately followed by its [cite:...] citation marker
   exactly as given. Do not modify quote text or citation markers.
3. Write supporting prose around the quotes — introduce them, analyze them,
   connect ideas between sections. Aim for clear, academic tone.
4. Return ONLY the markdown paper. No code fences, no commentary.

OUTLINE:
{outline}

QUOTES TO INCLUDE (exactly as given, with their citation markers):
{quotes_block}
"""
    resp = gemini_generate(prompt)
    return (resp.text or "").strip()


# ---------- UI ----------

st.set_page_config(page_title="SourceMatch", layout="wide")

# Wider sidebar so the workflow panel breathes.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { min-width: 440px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

ss = st.session_state
ss.setdefault("paper", "")
ss.setdefault("chunks", [])  # list[Chunk]
ss.setdefault("parsed_files", [])  # list[(name, pages)]
ss.setdefault("sections", [])  # list[dict]
ss.setdefault("quotes_by_section", {})  # dict[str, list[Quote]]
ss.setdefault("staged_urls", [])  # list[str]
ss.setdefault("bucket", [])  # list[Quote] — selected for full-paper draft


# ---------- Sidebar: SourceMatch workflow ----------

with st.sidebar:
    st.markdown("## SourceMatch")
    st.caption("Upload your outline. Upload your sources. Get real, verified quotes.")
    st.divider()

    # 1. Outline
    with st.expander("1. Outline", expanded=True):
        desc = st.text_input(
            "Describe your paper",
            placeholder="A paper on EU energy policy...",
            key="outline_desc",
        )
        if st.button("Generate outline with AI", use_container_width=True):
            if not desc.strip():
                st.warning("Enter a description first.")
            else:
                with st.spinner("Generating outline..."):
                    try:
                        outline = generate_outline_from_description(desc)
                        ss.paper = (outline + "\n\n" + ss.paper).strip()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not generate outline: {e}")
        st.caption("Or just write your outline at the top of your paper.")

    # 2. Sources
    with st.expander("2. Sources"):
        files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        # Enter-to-add URL form (st.form submits on Enter, no Cmd+Enter needed).
        with st.form("add_url_form", clear_on_submit=True):
            new_url = st.text_input(
                "Add a URL (press Enter)",
                placeholder="https://example.com/article",
            )
            submitted = st.form_submit_button("Add URL", use_container_width=True)
            if submitted and new_url.strip():
                ss.staged_urls.append(new_url.strip())
        if ss.staged_urls:
            st.caption("**URLs queued:**")
            for i, u in enumerate(ss.staged_urls):
                cols = st.columns([6, 1])
                with cols[0]:
                    st.caption(f"• {u}")
                with cols[1]:
                    if st.button("✕", key=f"rm_url_{i}"):
                        ss.staged_urls.pop(i)
                        st.rerun()

        if (files or ss.staged_urls) and st.button(
            "Parse sources", use_container_width=True
        ):
            ss.chunks = []
            ss.parsed_files = []
            next_id = 0
            with st.spinner("Parsing sources..."):
                for f in files or []:
                    try:
                        new_chunks = parse_pdf(f.getvalue(), f.name, next_id)
                        ss.chunks.extend(new_chunks)
                        next_id += len(new_chunks)
                        pages = max((c.page for c in new_chunks), default=0)
                        ss.parsed_files.append((f.name, pages))
                    except Exception as e:
                        st.error(f"Failed to parse {f.name}: {e}")
                for url in ss.staged_urls:
                    try:
                        new_chunks = parse_url(url, next_id)
                        if not new_chunks:
                            st.warning(f"No readable text at {url}")
                            continue
                        ss.chunks.extend(new_chunks)
                        next_id += len(new_chunks)
                        pages = max((c.page for c in new_chunks), default=0)
                        ss.parsed_files.append((new_chunks[0].doc, pages))
                    except Exception as e:
                        st.error(f"Failed to fetch {url}: {e}")
        if ss.parsed_files:
            st.success(
                f"Parsed {len(ss.parsed_files)} sources · "
                f"{len(ss.chunks)} passages indexed."
            )
            for name, pages in ss.parsed_files:
                st.caption(f"• {name} ({pages} pg)")

    # 3. Match
    with st.expander("3. Match"):
        ready = bool(ss.paper.strip()) and bool(ss.chunks)
        if st.button(
            "Find my quotes",
            type="primary",
            disabled=not ready,
            use_container_width=True,
        ):
            status = st.empty()
            status.markdown("Analyzing outline...")
            try:
                ss.sections = analyze_outline(ss.paper)
            except Exception as e:
                st.error(f"Outline analysis failed: {e}")
                ss.sections = []
            chunks_by_id = {c.id: c for c in ss.chunks}
            results: dict[str, list[Quote]] = {}
            total_dropped = 0
            n = max(1, len(ss.sections))
            prog = st.progress(0.0)
            for i, sec in enumerate(ss.sections):
                label = sec["section"][:40]
                status.markdown(f"Matching `{label}` ({i+1}/{n})...")
                try:
                    candidates = rank_quotes_for_section(
                        sec["section"], sec.get("evidence_need", ""), ss.chunks
                    )
                except Exception as e:
                    st.warning(f"{sec['section']}: {e}")
                    candidates = []
                verified = verify_quotes(candidates, chunks_by_id)
                for q in verified:
                    q.section = sec["section"]
                results[sec["section"]] = verified
                total_dropped += max(0, len(candidates) - len(verified))
                prog.progress((i + 1) / n)
            ss.quotes_by_section = results
            prog.empty()
            status.empty()
            kept = sum(len(v) for v in ss.quotes_by_section.values())
            if total_dropped:
                st.info(f"{kept} verified · {total_dropped} dropped (not verbatim).")
            else:
                st.success(f"{kept} verified quotes.")
        if not ready:
            st.caption(":gray[Add an outline and parse sources first.]")

    # 4. Review & Export
    with st.expander("4. Review & Export"):
        if not ss.quotes_by_section:
            st.caption(":gray[Run matching first.]")
        else:
            for section, quotes in ss.quotes_by_section.items():
                st.markdown(f"**{section}**")
                if not quotes:
                    st.caption("No verified quotes.")
                for i, q in enumerate(quotes):
                    preview = q.text if len(q.text) <= 90 else q.text[:90] + "…"
                    q.accepted = st.checkbox(
                        f"\"{preview}\"",
                        value=q.accepted,
                        key=f"acc_{section}_{i}",
                        help=f"{q.doc} · p. {q.page}",
                    )
                    in_bucket = any(
                        b.text == q.text and b.doc == q.doc and b.page == q.page
                        for b in ss.bucket
                    )
                    if st.button(
                        "✓ Added to draft" if in_bucket else "Add to draft",
                        key=f"ins_{section}_{i}",
                        use_container_width=True,
                        disabled=in_bucket,
                    ):
                        ss.bucket.append(q)
                        st.rerun()
            st.divider()
            for style in ("APA", "MLA", "Chicago"):
                st.download_button(
                    f"Download {style}",
                    data=build_export(ss.quotes_by_section, style),
                    file_name=f"sourcematch_{style.lower()}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"dl_{style}",
                )

    # 5. Write paper
    with st.expander(f"5. Write paper ({len(ss.bucket)} in draft)"):
        if not ss.bucket:
            st.caption(
                ":gray[No quotes selected yet — use **Add to draft** in panel 4.]"
            )
        else:
            for i, q in enumerate(ss.bucket):
                preview = q.text if len(q.text) <= 70 else q.text[:70] + "…"
                cols = st.columns([6, 1])
                with cols[0]:
                    st.caption(f"\"{preview}\"")
                    st.caption(f":gray[{q.section} · {q.doc} p. {q.page}]")
                with cols[1]:
                    if st.button("✕", key=f"rm_b_{i}"):
                        ss.bucket.pop(i)
                        st.rerun()
            st.divider()
            disabled = not ss.paper.strip()
            if st.button(
                "Write full paper",
                type="primary",
                use_container_width=True,
                disabled=disabled,
            ):
                with st.spinner("Drafting your paper..."):
                    try:
                        ss.paper = write_paper_from_bucket(ss.paper, ss.bucket)
                        st.success("Paper drafted. Open the Write or Preview tab.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Drafting failed: {e}")
            if disabled:
                st.caption(":gray[Add an outline first.]")

    st.divider()
    if st.button("Start over", use_container_width=True):
        for k in (
            "paper", "chunks", "parsed_files", "sections",
            "quotes_by_section", "staged_urls", "bucket",
        ):
            ss.pop(k, None)
        st.rerun()


# ---------- Main area: paper editor ----------

st.markdown(
    """
    <div style="padding: 8px 0 4px 0;">
      <h1 style="margin-bottom: 0;">Your paper</h1>
      <p style="color: #666; margin-top: 4px;">
        Write in markdown. Use <b>#</b> for headings, <b>**bold**</b>, <b>*italic*</b>,
        <b>-</b> for bullets. Switch to Preview to see it rendered.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

write_tab, preview_tab = st.tabs(["Write", "Preview"])

with write_tab:
    st.text_area(
        "Paper",
        height=700,
        label_visibility="collapsed",
        placeholder=(
            "# My Paper Title\n\n"
            "## I. Introduction\n\n"
            "Start writing here..."
        ),
        key="paper",
    )

with preview_tab:
    if ss.paper.strip():
        with st.container(border=True):
            st.markdown(
                render_paper_with_citations(ss.paper),
                unsafe_allow_html=True,
            )
    else:
        st.caption(":gray[Nothing to preview yet — write something in the Write tab.]")
