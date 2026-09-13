"""
AI PCB Engineering Reviewer
----------------------------
A Streamlit web app that lets a hardware engineer upload a schematic PDF
and get a deep, structured engineering review from a Google Gemini
multimodal model, focused on:
    1. Signal Integrity (SI)
    2. Power / Ground loop issues
    3. Common-mode coupling risks

How it works (high level):
    1. User uploads a schematic PDF.
    2. We render every page of the PDF into a high-resolution image
       (schematics are visual, so the AI needs to *see* them, not just
       read text).
    3. Each page image is base64-encoded and sent to the Gemini vision
       model along with a detailed engineering-review prompt.
    4. The model's markdown report is parsed into sections and shown
       in a clean, tabbed dashboard. The full report can be downloaded.

Deployment:
    - requirements.txt lists the needed packages.
    - Push app.py + requirements.txt to a GitHub repo and deploy on
      Streamlit Community Cloud (or any Streamlit-compatible host).
    - The user supplies their own Google Gemini API key at runtime, so no
      secret ever needs to be committed to the repo.

UI/UX Refactor Notes (v2):
    - Sidebar added: API key, model selection, and zoom slider moved here.
    - Native st.metric replaces custom metric-card HTML for better a11y.
    - Report panel wrapped in st.container(border=True) with status pill.
    - Tab content wrapped in st.container(border=True) for visual depth.
    - Download buttons grouped in a styled "download card" section.
    - Global CSS extended: sidebar polish, metric overrides, hover transitions,
      pulsing LED animation, download-card, and CSS custom-property tokens.
    - All backend logic, session_state, and API calls are unchanged.
"""

import io
import re

import fitz  # PyMuPDF - used to render PDF pages as images
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.6-flash"
MODEL_OPTIONS = [
    "gemini-3.6-flash",
]

REPORT_SECTIONS = [
    "Executive Summary",
    "Signal Integrity Analysis",
    "Power and Ground Loop Analysis",
    "Common Mode Coupling Analysis",
    "Prioritized Recommendations",
]


# ---------------------------------------------------------------------------
# PDF -> Image helpers
# ---------------------------------------------------------------------------

def render_pdf_pages_to_images(pdf_bytes: bytes, zoom: float = 2.0) -> list[Image.Image]:
    """Convert every page of an uploaded PDF into a PIL Image."""
    images = []
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix)
            img_bytes = pixmap.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)))
    return images


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_review_prompt(extra_notes: str) -> str:
    """Build the detailed engineering-review instruction sent to Gemini."""
    notes_block = f"\nAdditional design context from the user:\n{extra_notes}\n" if extra_notes else ""

    return f"""
You are a senior PCB hardware design engineer performing a rigorous schematic
review. You are shown one or more schematic pages (as images) from a single
PCB design. Study every net, component value, reference designator, and
connector you can identify.
{notes_block}
Produce a deep-dive engineering report in MARKDOWN using EXACTLY these five
'##' headings, in this order, and nothing else before or after them:

## Executive Summary
A short (4-6 sentence) plain-language overview of the design's overall
health and the single biggest risk found.

## Signal Integrity Analysis
Identify and explain concerns such as: high-speed / clock / differential
pairs lacking series or termination resistors, impedance-sensitive nets
without a clear reference plane, long unbuffered traces implied by the
schematic, missing decoupling near high-speed ICs, fan-out or routing
choices visible in the schematic that risk reflections or crosstalk,
and any single-ended signals that should likely be differential.

## Power and Ground Loop Analysis
Identify and explain concerns such as: decoupling capacitor placement and
values versus IC power pins, missing bulk/bypass capacitance, star-grounding
vs multi-point grounding conflicts, ground return path length for
high-current or high-speed loops, split/isolated ground schemes and their
stitching, power sequencing risks, and any large physical loop areas
implied by how power and return nets are drawn.

## Common Mode Coupling Analysis
Identify and explain concerns such as: cable/connector shield grounding,
common-mode choke usage (or absence) on I/O and power lines, isolation
barrier crossings, unbalanced differential routing that could convert
differential noise to common-mode, and proximity of noisy switching nets
to sensitive analog or shield references.

## Prioritized Recommendations
A numbered list (highest impact first) of concrete, actionable fixes. For
each item state: the issue, the risk if unresolved, and the specific
schematic-level fix (e.g. component to add, net to reroute, value to
change).

Formatting rules:
- Reference specific component designators, net names, or page numbers
  whenever you can see them in the image.
- If a page's image quality or resolution prevents you from confirming a
  detail, say so explicitly rather than guessing silently.
- Be direct and technical; this report is for a hardware engineer, not a
  general audience.
""".strip()


def build_gemini_contents(images: list[Image.Image], extra_notes: str) -> list:
    """Build Gemini multimodal contents from rendered schematic pages."""
    contents = []
    for img in images:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        contents.append(
            types.Part.from_bytes(
                data=buffer.getvalue(),
                mime_type="image/png",
            )
        )
    contents.append(build_review_prompt(extra_notes))
    return contents


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------

def call_gemini(api_key: str, model: str, contents: list) -> str:
    """Send the multimodal PCB review request to Google Gemini."""
    if not api_key:
        raise RuntimeError("Gemini API key is missing.")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an expert PCB and signal-integrity engineer with "
                    "20+ years of hardware review experience. Give precise, "
                    "actionable, technically grounded feedback. Analyze only "
                    "what can reasonably be established from the supplied "
                    "schematic images and clearly mark uncertainty."
                ),
                temperature=0.2,
                max_output_tokens=6000,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        return response.text
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Report parsing / display helpers
# ---------------------------------------------------------------------------

def split_report_into_sections(report_text: str) -> dict[str, str]:
    """Split a markdown report into a dict keyed by REPORT_SECTIONS titles."""
    sections: dict[str, str] = {}
    parts = re.split(r"\n(?=##\s+)", report_text.strip())
    for part in parts:
        match = re.match(r"##\s+(.*?)\n(.*)", part.strip(), re.DOTALL)
        if match:
            title, body = match.group(1).strip(), match.group(2).strip()
            sections[title] = body
        elif part.strip():
            sections.setdefault("Other Notes", "")
            sections["Other Notes"] += part.strip() + "\n"
    return sections


# ---------------------------------------------------------------------------
# Streamlit UI helpers
# ---------------------------------------------------------------------------

from io import BytesIO
from xml.sax.saxutils import escape


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

/* ── Design tokens ───────────────────────────────────────────────────── */
:root {
    --bg:           #07110d;
    --surface:      #0d1813;
    --surface-2:    #111f18;
    --surface-3:    #16271e;
    --primary:      #65d46e;
    --primary-soft: #9be6a0;
    --accent:       #2fbf71;
    --text:         #eef7f0;
    --muted:        #9eaea3;
    --border:       rgba(101, 212, 110, .20);
    --border-soft:  rgba(255,255,255,.08);
    --radius-lg:    16px;
    --radius-md:    12px;
    --radius-sm:    8px;
}

html, body, [class*="css"] { font-family: Inter, sans-serif; color: var(--text); }

.stApp {
    background:
        radial-gradient(circle at 50% -8%, rgba(47,191,113,.12), transparent 34%),
        linear-gradient(180deg, #08120e 0%, #050a07 100%);
}

.block-container { max-width: 1320px; padding: 2.4rem 2rem 4.5rem; }

/* ── Hero ────────────────────────────────────────────────────────────── */
.pcb-board {
    position: relative; overflow: hidden;
    border: 1px solid var(--border); border-radius: 22px;
    padding: 2.35rem 2.5rem; margin-bottom: 1.35rem;
    background: linear-gradient(135deg, rgba(47,191,113,.10), rgba(13,24,19,.94) 52%), #0a140f;
    box-shadow: 0 18px 60px rgba(0,0,0,.30);
}

.pcb-board:before {
    content: ""; position: absolute; inset: 0; opacity: .23;
    background:
        linear-gradient(90deg, transparent 49.7%, rgba(101,212,110,.16) 50%, transparent 50.3%),
        linear-gradient(transparent 49.7%, rgba(101,212,110,.12) 50%, transparent 50.3%);
    background-size: 46px 46px; pointer-events: none;
}

.eyebrow {
    position: relative; color: var(--primary);
    font-family: "JetBrains Mono", monospace;
    font-size: .72rem; font-weight: 600; letter-spacing: .13em; text-transform: uppercase;
}

.pcb-board h1 {
    position: relative;
    font-size: clamp(2.35rem, 4.4vw, 4rem); line-height: 1.04;
    letter-spacing: -.045em; font-weight: 800; margin: .65rem 0 .85rem; color: #f7fbf8;
}

.pcb-board h1 span { color: var(--primary); }

.hero-copy { position: relative; max-width: 760px; color: #b4c2b8; font-size: .98rem; line-height: 1.7; }

/* Online status pill */
.trace-status {
    position: relative; display: inline-flex; align-items: center; gap: .5rem;
    padding: .42rem .72rem; margin-top: 1rem;
    border: 1px solid rgba(101,212,110,.25); border-radius: 999px;
    background: rgba(101,212,110,.055); color: var(--primary-soft);
    font-family: "JetBrains Mono", monospace; font-size: .68rem; font-weight: 600; letter-spacing: .04em;
}

/* Pulsing LED — v2: animation added */
.led {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--primary); box-shadow: 0 0 10px rgba(101,212,110,.65);
    animation: led-pulse 2.2s ease-in-out infinite;
}

@keyframes led-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .4; } }

/* ── Workflow steps ───────────────────────────────────────────────────── */
.workflow { display: grid; grid-template-columns: 1fr 1fr; gap: .9rem; margin: 0 0 1.35rem; }

.step {
    border: 1px solid var(--border-soft); border-radius: 15px;
    padding: .95rem 1.05rem; background: rgba(13,24,19,.78);
}

.step-num {
    display: inline-grid; place-items: center;
    width: 28px; height: 28px; margin-right: .55rem; border-radius: 50%;
    background: var(--primary); color: #07110d; font-size: .78rem; font-weight: 800;
}

.step b { color: #f0f6f1; font-size: .9rem; }
.step small { display: block; margin-top: .35rem; color: var(--muted); font-size: .75rem; line-height: 1.5; }

/* ── Feature cards — v2: hover transitions ───────────────────────────── */
.feature-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem; margin: 0 0 1.65rem; }

.feature {
    min-height: 125px; border: 1px solid var(--border-soft);
    border-radius: var(--radius-lg); padding: 1.15rem;
    background: linear-gradient(145deg, rgba(17,31,24,.92), rgba(9,17,13,.94));
    box-shadow: 0 8px 25px rgba(0,0,0,.12);
    transition: border-color .2s ease, box-shadow .2s ease;
}

.feature:hover { border-color: var(--border); box-shadow: 0 12px 35px rgba(47,191,113,.10); }

.feature .icon { font-size: 1.3rem; }
.feature b { display: block; margin: .42rem 0 .28rem; color: var(--primary-soft); font-size: .88rem; font-weight: 700; }
.feature span { color: var(--muted); font-size: .75rem; line-height: 1.55; }

/* ── Section labels ──────────────────────────────────────────────────── */
.section-label {
    color: var(--primary); font-family: "JetBrains Mono", monospace;
    font-size: .68rem; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; margin: 1.5rem 0 .7rem;
}

/* ── File uploader ───────────────────────────────────────────────────── */
.upload-shell {
    border: 1px dashed rgba(101,212,110,.36); border-radius: var(--radius-lg);
    padding: .65rem; background: rgba(101,212,110,.025);
}

[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #173a28 0%, #10261c 55%, #0c1812 100%) !important;
    border-radius: 13px !important; border: 1.5px solid rgba(101,212,110,.62) !important;
    box-shadow: inset 0 0 30px rgba(101,212,110,.045), 0 8px 26px rgba(0,0,0,.18) !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #9be6a0 !important;
    box-shadow: inset 0 0 34px rgba(101,212,110,.08), 0 10px 30px rgba(47,191,113,.12) !important;
}

[data-testid="stFileUploaderDropzone"] * { color: #eef7f0 !important; font-size: .84rem !important; font-weight: 600 !important; }

[data-testid="stFileUploaderDropzone"] button {
    background: #65d46e !important; color: #07110d !important;
    border: 1px solid #65d46e !important; border-radius: 9px !important; font-weight: 800 !important;
}

[data-testid="stFileUploaderDropzone"] button:hover { background: #9be6a0 !important; border-color: #9be6a0 !important; }

/* ── Inputs / textareas ──────────────────────────────────────────────── */
textarea, input {
    background: #3d4741 !important; color: #f4f8f5 !important;
    border: 1px solid rgba(101,212,110,.28) !important;
    border-radius: 10px !important; font-size: .86rem !important;
}

div[data-testid="stTextArea"] label p {
    color: #f1f7f2 !important; font-size: .98rem !important;
    font-weight: 800 !important; letter-spacing: -.01em !important;
}

div[data-testid="stTextArea"] label p:before { content: "✦ "; color: var(--primary); }

div[data-testid="stTextArea"] { padding: .15rem; border-radius: 13px; }

div[data-testid="stTextArea"] > div { border-radius: 12px; box-shadow: 0 0 0 1px rgba(101,212,110,.05); }

textarea::placeholder, input::placeholder { color: #c2ccc5 !important; opacity: 1 !important; }

/* ── Generic card base ───────────────────────────────────────────────── */
.status-card, .metric-card, .report-shell {
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-lg);
    background: rgba(12,22,17,.90);
}

.status-card { padding: 1.05rem 1.15rem; }

/* Legacy custom metric card (kept for render_metric() backward compat) */
.metric-card { padding: .9rem 1rem; min-height: 72px; }
.metric-number { font-size: 1.35rem; line-height: 1.1; font-weight: 800; color: var(--primary); }
.metric-label { margin-top: .28rem; color: var(--muted); font-size: .72rem; }

/* ── Native st.metric card — v2 ──────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(12,22,17,.90) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: var(--radius-md) !important;
    padding: .85rem 1rem !important;
    box-shadow: 0 4px 16px rgba(0,0,0,.14);
    transition: box-shadow .2s ease;
}

[data-testid="stMetric"]:hover { box-shadow: 0 6px 22px rgba(47,191,113,.10); }

[data-testid="stMetricLabel"] {
    color: var(--muted) !important; font-size: .72rem !important;
    font-weight: 600 !important; letter-spacing: .04em !important;
}

[data-testid="stMetricValue"] { color: var(--primary) !important; font-size: 1.55rem !important; font-weight: 800 !important; }

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button {
    background: var(--primary) !important; color: #07110d !important;
    border: 1px solid var(--primary) !important; border-radius: 10px !important;
    font-size: .84rem !important; font-weight: 800 !important; min-height: 2.75rem;
    box-shadow: 0 7px 22px rgba(47,191,113,.12); transition: all .18s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: #82df88 !important; border-color: #82df88 !important;
    color: #061009 !important; transform: translateY(-1px);
    box-shadow: 0 10px 28px rgba(47,191,113,.22);
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: .35rem; }

.stTabs [data-baseweb="tab"] {
    border-radius: 9px; padding: .55rem .8rem;
    background: #0e1812; color: #aebbb1; font-size: .78rem; font-weight: 600;
}

.stTabs [aria-selected="true"] { background: rgba(101,212,110,.09); color: var(--primary) !important; }

.report-shell { padding: 1.2rem; }

.report-title { font-size: 1.45rem; font-weight: 800; letter-spacing: -.025em; color: #f4f8f5; margin-bottom: .25rem; }

.report-subtitle {
    color: var(--muted); font-family: "JetBrains Mono", monospace;
    font-size: .66rem; letter-spacing: .04em;
}

/* ── Download card — v2 ──────────────────────────────────────────────── */
.download-card {
    border: 1px solid var(--border-soft); border-radius: var(--radius-lg);
    background: rgba(12,22,17,.90); padding: 1.2rem 1.4rem; margin-top: .5rem;
}

.download-card-title {
    font-size: .78rem; font-weight: 700; color: var(--primary-soft);
    font-family: "JetBrains Mono", monospace;
    letter-spacing: .08em; text-transform: uppercase; margin-bottom: .75rem;
}

/* ── Sidebar polish — v2 ──────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border-soft) !important; }

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text) !important; font-size: .83rem !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text) !important;
}

.sidebar-section-label {
    color: var(--primary); font-family: "JetBrains Mono", monospace;
    font-size: .65rem; font-weight: 600; letter-spacing: .13em;
    text-transform: uppercase; margin: 1.2rem 0 .5rem;
}

.small-note { color: var(--muted); font-size: .72rem; line-height: 1.55; }

@media (max-width: 800px) {
    .feature-row, .workflow { grid-template-columns: 1fr; }
    .pcb-board { padding: 1.55rem; }
    .block-container { padding: 1.15rem 1rem 3rem; }
    .pcb-board h1 { font-size: 2.35rem; }
}
</style>
"""


def configure_page():
    """Configure page settings and inject the global CSS design system."""
    st.set_page_config(
        page_title="PCB/Schematic Reviewer",
        page_icon="🟩",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


def render_metric(number: str, label: str):
    """Render a legacy custom metric card (kept for backward compatibility)."""
    st.markdown(
        f'<div class="metric-card"><div class="metric-number">{number}</div><div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def make_pdf(report_text: str, filename: str = "pcb_schematic_review.pdf") -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    from reportlab.lib.units import mm

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("PCBTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, textColor=colors.HexColor("#111111"), spaceAfter=10)
    body = ParagraphStyle("PCBBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=14, textColor=colors.HexColor("#222222"), spaceAfter=7)
    heading = ParagraphStyle("PCBHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, textColor=colors.HexColor("#111111"), spaceBefore=10, spaceAfter=6)

    story = [Paragraph("PCB/Schematic Reviewer — Engineering Report", title),
             Paragraph("AI-assisted engineering review", body),
             Spacer(1, 5)]

    for raw in report_text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 4))
        elif line.startswith("#"):
            text = re.sub(r"^#+\s*", "", line)
            story.append(Paragraph(escape(text), heading))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph("• " + escape(line[2:]), body))
        else:
            story.append(Paragraph(escape(line), body))

    doc.build(story)
    return buf.getvalue()


def make_docx(report_text: str) -> bytes:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_heading("PCB/Schematic Reviewer — Engineering Report", level=0)
    p = doc.add_paragraph("AI-assisted engineering review")
    p.runs[0].bold = True

    for raw in report_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 3)
            doc.add_heading(re.sub(r"^#+\s*", "", line), level=level)
        elif line.startswith(("- ", "* ")):
            doc.add_paragraph(line[2:], style="List Bullet")
        else:
            doc.add_paragraph(line)

    for p in doc.paragraphs:
        for run in p.runs:
            run.font.name = "Arial"
            run.font.size = Pt(10)

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def render_empty_state():
    """Placeholder card shown before any schematic is uploaded."""
    st.markdown(
        """
        <div class="status-card" style="text-align:center; padding:2.4rem 1.5rem;">
            <div style="font-size:2.5rem;">🟩</div>
            <h3 style="margin:.5rem 0; color:#fff;">Your engineering review starts here</h3>
            <div style="color:#8b978d; max-width:650px; margin:auto;">
                Add a schematic PDF, optionally tell us about the design, and let the reviewer
                surface the highest-value engineering risks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar — settings panel (UI improvement: moved from inline to sidebar)
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """
    Render the settings sidebar and return a settings dict.

    UI change: configuration previously hardcoded inline is now collected
    in a clean, organised sidebar with section labels and helper captions.
    Backend values (api_key, model, zoom) are identical to the original.
    """
    with st.sidebar:
        # ── Brand mark ────────────────────────────────────────────────
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:.6rem;padding:.4rem 0 1rem;">
                <div style="font-size:1.5rem;">🟩</div>
                <div>
                    <div style="font-weight:800;font-size:.95rem;color:#f0f6f1;">PCB Reviewer</div>
                    <div style="font-size:.68rem;color:#7a8f80;font-family:'JetBrains Mono',monospace;letter-spacing:.06em;">AI ENGINEERING ANALYSIS</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── API configuration ─────────────────────────────────────────
        st.markdown('<div class="sidebar-section-label">⚙ Configuration</div>', unsafe_allow_html=True)

        secret_key = st.secrets.get("GEMINI_API_KEY", None)
        if secret_key:
            api_key = secret_key
            st.caption("✅ API key loaded from Streamlit Secrets.")
        else:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIza…",
                help="Paste your Google AI Studio API key. It is never stored.",
            )

        model = st.selectbox(
            "Model",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(DEFAULT_MODEL),
            help="Choose the Gemini model used for the review.",
        )

        st.divider()

        # ── Render quality ────────────────────────────────────────────
        st.markdown('<div class="sidebar-section-label">🔬 Render quality</div>', unsafe_allow_html=True)

        zoom = st.slider(
            "PDF zoom factor",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.5,
            help="Higher values → sharper images sent to the AI, but slower processing.",
        )
        st.caption(f"Effective resolution: **{int(72 * zoom)} DPI** — {'High detail' if zoom >= 2.0 else 'Standard detail'}")

        st.divider()

        # ── About ─────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section-label">ℹ About</div>', unsafe_allow_html=True)
        st.caption(
            "Analyzes schematic PDFs for **signal integrity**, **power & ground loops**, "
            "and **EMI / common-mode coupling** using Google Gemini's multimodal vision."
        )

    return {"api_key": api_key, "model": model, "zoom": zoom}


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    configure_page()

    # ── Settings sidebar ──────────────────────────────────────────────────
    # UI improvement: sidebar collects all configuration so the main canvas
    # stays focused on upload → analyze → report.
    settings = render_sidebar()

    # ── Hero banner ───────────────────────────────────────────────────────
    st.markdown(
        """
        <section class="pcb-board">
            <div class="eyebrow">PCB / SCHEMATIC REVIEWER</div>
            <h1>Review your design.<br><span>Catch risks earlier.</span></h1>
            <div class="hero-copy">
                AI-assisted engineering analysis for signal integrity, power &amp; ground,
                and EMI/common-mode coupling — presented as practical, prioritized findings.
            </div>
            <div class="trace-status"><span class="led"></span> ANALYSIS CONSOLE ONLINE</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── How it works ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="workflow">
            <div class="step"><span class="step-num">1</span><b>Add schematic</b><small>Upload your PDF and optionally add design context.</small></div>
            <div class="step"><span class="step-num">2</span><b>Analyze PCB</b><small>Run the AI review and inspect prioritized engineering findings.</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Capability feature cards ───────────────────────────────────────────
    st.markdown(
        """
        <div class="feature-row">
            <div class="feature"><div class="icon">⚡</div><b>Signal Integrity</b><span>Termination, reflections, interfaces and high-speed risk areas.</span></div>
            <div class="feature"><div class="icon">🔋</div><b>Power &amp; Ground</b><span>Decoupling, return paths, loops and power integrity concerns.</span></div>
            <div class="feature"><div class="icon">📡</div><b>EMI &amp; Coupling</b><span>Common-mode paths, filtering, shields and coupling risks.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 01 — Upload + Design context ─────────────────────────────
    st.markdown('<div class="section-label">01 · Add schematic</div>', unsafe_allow_html=True)

    upload_col, context_col = st.columns([1.3, 1], gap="large")

    with upload_col:
        # UI improvement: uploader wrapped in a styled dashed-border shell
        st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
        uploaded_pdf = st.file_uploader(
            "📄 Drag & drop your schematic PDF",
            type=["pdf"],
            help="Multi-page schematic PDFs are supported.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_pdf:
            st.success(f"✓ {uploaded_pdf.name} loaded • {uploaded_pdf.size / 1024:.0f} KB", icon="🟩")

    with context_col:
        st.markdown(
            '<div style="font-size:.98rem;font-weight:800;color:#f1f7f2;margin:0 0 .35rem;">✦ Design context <span style="color:#9eaea3;font-size:.78rem;font-weight:600;">(optional)</span></div>',
            unsafe_allow_html=True,
        )
        extra_notes = st.text_area(
            "Design context (optional)",
            placeholder="Example: 4-layer board • USB 3.x • buck regulator on page 2 • sensitive ADC on page 4",
            height=132,
            label_visibility="collapsed",
        )

    # ── Section 02 — Analyze (only visible after upload) ─────────────────
    if uploaded_pdf:
        st.markdown('<div class="section-label">02 · Analyze PCB</div>', unsafe_allow_html=True)

        # UI improvement: native st.metric inside CSS-styled containers
        # replaces the custom render_metric HTML helper for better a11y.
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Schematic", value="✓ Ready")
        with m2:
            st.metric(label="Render quality", value=f"{settings['zoom']:.1f}×")
        with m3:
            st.metric(label="Review mode", value="AI")

        st.write("")
        run_clicked = st.button(
            "⚡ Analyze My PCB/Schematic",
            type="primary",
            use_container_width=True,
        )
    else:
        run_clicked = False
        render_empty_state()

    # ── Run analysis ──────────────────────────────────────────────────────
    if run_clicked:
        if not settings["api_key"]:
            st.error("GEMINI_API_KEY is missing. Add it to Streamlit Secrets or paste it in the sidebar.")
            return

        pdf_bytes = uploaded_pdf.getvalue()

        with st.status("Preparing schematic…", expanded=True) as status:
            st.write("Rendering schematic pages.")
            images = render_pdf_pages_to_images(pdf_bytes, zoom=settings["zoom"])
            st.write(f"✓ {len(images)} page(s) prepared")
            status.update(label="Schematic ready", state="complete", expanded=False)

        with st.expander(f"👁 Preview submitted schematic • {len(images)} page(s)", expanded=False):
            cols = st.columns(min(len(images), 4) or 1)
            for i, img in enumerate(images):
                cols[i % len(cols)].image(img, caption=f"Page {i + 1}", use_container_width=True)

        try:
            with st.status("🤖 Analyzing your PCB/schematic…", expanded=True) as status:
                st.write("Checking signal integrity, power/ground, and EMI & coupling.")
                contents = build_gemini_contents(images, extra_notes)
                report_text = call_gemini(settings["api_key"], settings["model"], contents)
                status.update(label="Analysis complete", state="complete", expanded=False)
        except RuntimeError as exc:
            st.error(f"Analysis failed: {exc}")
            return

        st.session_state["last_report"] = report_text

    # ── Report display ────────────────────────────────────────────────────
    if "last_report" in st.session_state:
        st.markdown('<div class="section-label">Engineering report</div>', unsafe_allow_html=True)

        # UI improvement: report header uses a native bordered container
        # with an inline status pill instead of a bare HTML div block.
        with st.container(border=True):
            hdr_left, hdr_right = st.columns([3, 1])
            with hdr_left:
                st.markdown(
                    """
                    <div class="report-title">PCB/Schematic Review Report</div>
                    <div class="report-subtitle">AI-ASSISTED &nbsp;·&nbsp; PRIORITIZED ENGINEERING FINDINGS</div>
                    """,
                    unsafe_allow_html=True,
                )
            with hdr_right:
                st.markdown(
                    '<div style="text-align:right;padding-top:.35rem;"><span class="trace-status"><span class="led"></span> REPORT READY</span></div>',
                    unsafe_allow_html=True,
                )

        st.write("")

        sections = split_report_into_sections(st.session_state["last_report"])
        tab_titles = [t for t in REPORT_SECTIONS if t in sections] + [t for t in sections if t not in REPORT_SECTIONS]
        tabs = st.tabs([f"  {t}" for t in tab_titles])

        for tab, title in zip(tabs, tab_titles):
            with tab:
                # UI improvement: each tab body wrapped in a bordered container
                # for visual separation from the tab bar chrome.
                with st.container(border=True):
                    st.markdown(sections[title])

        # ── Download card ─────────────────────────────────────────────
        # UI improvement: download buttons grouped in a clearly labelled
        # card instead of floating at the bottom of the page.
        st.write("")
        st.markdown('<div class="download-card">', unsafe_allow_html=True)
        st.markdown('<div class="download-card-title">⬇ Export report</div>', unsafe_allow_html=True)

        pdf_bytes_dl = make_pdf(st.session_state["last_report"])
        docx_bytes_dl = make_docx(st.session_state["last_report"])

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Download PDF report",
                data=pdf_bytes_dl,
                file_name="pcb_schematic_review.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇️ Download Word report",
                data=docx_bytes_dl,
                file_name="pcb_schematic_review.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Reports contain the complete raw AI-generated analysis. Review with a qualified hardware engineer before acting on findings.")


if __name__ == "__main__":
    main()
