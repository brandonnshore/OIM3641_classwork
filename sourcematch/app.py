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


# ---------- UI ----------

st.set_page_config(page_title="SourceMatch", layout="wide")

# session state defaults
ss = st.session_state
ss.setdefault("step", 1)
ss.setdefault("outline", "")
ss.setdefault("chunks", [])  # list[Chunk]
ss.setdefault("sections", [])  # list[dict]
ss.setdefault("quotes_by_section", {})  # dict[str, list[Quote]]
ss.setdefault("parsed_files", [])  # list[(name, pages)]

# header
st.markdown(
    """
    <div style="text-align:center; padding: 12px 0 4px 0;">
      <h1 style="margin-bottom: 0;">SourceMatch</h1>
      <p style="color: #666; margin-top: 4px;">
        Upload your outline. Upload your sources. Get real, verified quotes.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# stepper
cols = st.columns(4)
labels = ["1. Outline", "2. Sources", "3. Match", "4. Review and Export"]
for i, (c, label) in enumerate(zip(cols, labels), start=1):
    with c:
        if i == ss.step:
            st.markdown(f"**{label}**")
        elif i < ss.step:
            st.markdown(f"**{label}** (done)")
        else:
            st.markdown(f":gray[{label}]")
st.divider()


# ---------- Step 1: Outline ----------

if ss.step == 1:
    st.subheader("Paper outline")
    st.caption(
        "Start by providing your paper outline. "
        "This tells us what quotes to look for in your sources."
    )

    ss.outline = st.text_area(
        "Outline",
        value=ss.outline,
        height=260,
        placeholder=(
            "Renewable Energy Policy in the EU\n\n"
            "I. Introduction — current state of EU energy policy\n"
            "II. Literature Review — effectiveness studies\n"
            "III. Policy Analysis — comparing member states\n"
            "IV. Conclusion — recommendations"
        ),
    )

    with st.expander("Generate outline with AI"):
        desc = st.text_input(
            "Short description of your paper",
            placeholder="A paper on how EU energy policy has shaped renewable adoption.",
        )
        if st.button("Generate with AI", type="secondary"):
            if not desc.strip():
                st.warning("Enter a description first.")
            else:
                with st.spinner("Generating outline..."):
                    try:
                        ss.outline = generate_outline_from_description(desc)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not generate outline: {e}")

    st.write("")
    disabled = len(ss.outline.strip()) < 20
    if st.button("Next: upload sources", type="primary", disabled=disabled):
        ss.step = 2
        st.rerun()
    if disabled:
        st.caption(":gray[Complete outline to continue]")


# ---------- Step 2: Sources ----------

elif ss.step == 2:
    st.success(f"Outline ready — {ss.outline.splitlines()[0][:80]}")
    st.subheader("Research sources")
    st.caption(
        "Drop PDF files. A Python parser extracts text deterministically — "
        "every quote will trace back to a specific page."
    )

    files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if files:
        if st.button("Parse uploaded files"):
            ss.chunks = []
            ss.parsed_files = []
            next_id = 0
            prog = st.progress(0.0)
            for i, f in enumerate(files):
                try:
                    new_chunks = parse_pdf(f.getvalue(), f.name, next_id)
                    ss.chunks.extend(new_chunks)
                    next_id += len(new_chunks)
                    # use a rough page count = max page number seen
                    pages = max((c.page for c in new_chunks), default=0)
                    ss.parsed_files.append((f.name, pages))
                except Exception as e:
                    st.error(f"Failed to parse {f.name}: {e}")
                prog.progress((i + 1) / len(files))
            prog.empty()

    if ss.parsed_files:
        total_pages = sum(p for _, p in ss.parsed_files)
        st.write(
            f"**{len(ss.parsed_files)} sources parsed · "
            f"{total_pages} pages · {len(ss.chunks)} passages indexed**"
        )
        for name, pages in ss.parsed_files:
            st.write(f"{name}  —  {pages} pg  (parsed)")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Back", key="back_2"):
            ss.step = 1
            st.rerun()
    with c2:
        if st.button(
            "Find my quotes",
            type="primary",
            disabled=len(ss.chunks) == 0,
        ):
            ss.step = 3
            st.rerun()


# ---------- Step 3: Matching ----------

elif ss.step == 3:
    st.subheader("Matching quotes to outline sections...")
    st.caption(
        f"Scanning {len(ss.parsed_files)} sources · "
        f"{sum(p for _, p in ss.parsed_files)} pages"
    )

    if not ss.quotes_by_section:
        steps_box = st.container()
        with steps_box:
            s1 = st.empty()
            s2 = st.empty()
            s3 = st.empty()
            s4 = st.empty()

        s1.markdown("Step 1. Analyzing outline for evidence needs...")
        try:
            ss.sections = analyze_outline(ss.outline)
        except Exception as e:
            st.error(f"Outline analysis failed: {e}")
            st.stop()
        s1.markdown("Step 1. Outline analyzed.")

        s2.markdown("Step 2. Searching across parsed passages...")
        chunks_by_id = {c.id: c for c in ss.chunks}
        s2.markdown("Step 2. Passages indexed.")

        s3.markdown("Step 3. Ranking best-fitting quotes per section...")
        all_verified: dict[str, list[Quote]] = {}
        total_candidates = 0
        total_dropped = 0
        drop_report: list[tuple[str, int, int]] = []  # (section, kept, dropped)
        prog = st.progress(0.0)
        for i, sec in enumerate(ss.sections):
            try:
                candidates = rank_quotes_for_section(
                    sec["section"], sec.get("evidence_need", ""), ss.chunks
                )
            except Exception as e:
                st.warning(f"Ranking failed for {sec['section']}: {e}")
                candidates = []
            verified = verify_quotes(candidates, chunks_by_id)
            for q in verified:
                q.section = sec["section"]
            all_verified[sec["section"]] = verified
            kept = len(verified)
            dropped = max(0, len(candidates) - kept)
            total_candidates += len(candidates)
            total_dropped += dropped
            drop_report.append((sec["section"], kept, dropped))
            prog.progress((i + 1) / max(1, len(ss.sections)))
        prog.empty()
        s3.markdown("Step 3. Quotes ranked.")

        s4.markdown("Step 4. :red[Verification — exact-match against original source text]")
        ss.quotes_by_section = all_verified
        total = sum(len(v) for v in all_verified.values())

        # Report: candidates returned vs quotes that survived
        # Python's exact-substring verification.
        if total_dropped > 0:
            s4.markdown(
                f"Step 4. {total_dropped} unverifiable quote(s) auto-dropped. "
                f"{total} verified against source documents."
            )
            st.warning(
                f"Verification guard removed {total_dropped} of "
                f"{total_candidates} candidate quotes that did not appear "
                f"verbatim in the source text."
            )
        else:
            s4.markdown(
                f"Step 4. {total} quotes verified verbatim against source documents."
            )
            st.success("All quotes verified against source documents.")

        with st.expander("Verification details"):
            for name, kept, dropped in drop_report:
                tag = "DROPPED" if dropped else "OK"
                st.write(
                    f"[{tag}] **{name}** — {kept} kept, {dropped} dropped"
                )

    total = sum(len(v) for v in ss.quotes_by_section.values())
    st.write(f"**{total} verified quotes** across {len(ss.parsed_files)} sources.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="back_3"):
            ss.step = 2
            st.rerun()
    with c2:
        if st.button("Review and export", type="primary"):
            ss.step = 4
            st.rerun()


# ---------- Step 4: Review & Export ----------

elif ss.step == 4:
    total = sum(len(v) for v in ss.quotes_by_section.values())
    st.subheader(f"{total} quotes found")
    st.caption(
        f"{total} verified quotes found across {len(ss.parsed_files)} sources. "
        "All extracted directly from your documents."
    )

    if not ss.quotes_by_section:
        st.info("No quotes yet — run matching first.")
        if st.button("Back to matching"):
            ss.step = 3
            st.rerun()
        st.stop()

    left, right = st.columns([1, 3])

    with left:
        st.markdown("**OUTLINE**")
        section_names = list(ss.quotes_by_section.keys())
        selected = st.radio(
            "Sections",
            section_names,
            label_visibility="collapsed",
        )

    with right:
        st.markdown(f"### {selected}")
        quotes = ss.quotes_by_section.get(selected, [])
        if not quotes:
            st.info("No verified quotes for this section.")
        for i, q in enumerate(quotes):
            with st.container(border=True):
                st.markdown(f"*\"{q.text}\"*")
                st.caption(f"{q.doc} · p. {q.page}")
                q.accepted = st.checkbox(
                    "Accept",
                    value=q.accepted,
                    key=f"acc_{selected}_{i}",
                )

    st.divider()
    st.markdown("**Export citations**")
    ec1, ec2, ec3, ec4 = st.columns([1, 1, 1, 3])
    for col, style in zip((ec1, ec2, ec3), ("APA", "MLA", "Chicago")):
        with col:
            text = build_export(ss.quotes_by_section, style)
            st.download_button(
                f"Download {style}",
                data=text,
                file_name=f"sourcematch_{style.lower()}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.write("")
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("Back", key="back_4"):
            ss.step = 3
            st.rerun()
    with bc2:
        if st.button("Start over"):
            for k in [
                "step",
                "outline",
                "chunks",
                "sections",
                "quotes_by_section",
                "parsed_files",
            ]:
                ss.pop(k, None)
            st.rerun()
