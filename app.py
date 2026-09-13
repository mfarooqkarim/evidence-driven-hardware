"""
AI PCB Engineering Reviewer
----------------------------
Streamlit application for reviewing uploaded schematic PDFs with Gemini.

Main improvements:
- High-contrast engineering report styling
- Solid report surfaces instead of translucent backgrounds
- Brighter headings, body text, lists, tables, and tabs
- Better structured AI recommendations
- Preserves the original Gemini/PDF/Word download workflow
"""

import io
import re
from io import BytesIO
from xml.sax.saxutils import escape

import fitz  # PyMuPDF
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

def render_pdf_pages_to_images(
    pdf_bytes: bytes,
    zoom: float = 2.0,
) -> list[Image.Image]:
    """Convert every page of an uploaded PDF into a PIL Image."""
    images = []
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix)
            img_bytes = pixmap.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    return images


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_review_prompt(extra_notes: str) -> str:
    """Build the engineering-review instruction sent to Gemini."""

    notes_block = (
        f"\nAdditional design context from the user:\n{extra_notes}\n"
        if extra_notes
        else ""
    )

    return f"""
You are a senior PCB hardware design engineer performing a rigorous schematic
review. You are shown one or more schematic pages from a single PCB design.
Study every net, component value, reference designator, connector, and page
detail you can identify.

{notes_block}

Produce a deep-dive engineering report in MARKDOWN using EXACTLY these five
'##' headings, in this order, and nothing else before or after them:

## Executive Summary
Write a short 4-6 sentence plain-language overview of the design's overall
health and the single biggest risk found. Clearly distinguish confirmed
observations from assumptions and missing information.

## Signal Integrity Analysis
Identify and explain concerns such as: high-speed, clock, or differential
pairs lacking series or termination resistors; impedance-sensitive nets
without a clear reference plane; long unbuffered traces implied by the
schematic; missing decoupling near high-speed ICs; fan-out or routing choices
visible in the schematic that risk reflections or crosstalk; and single-ended
signals that may need differential treatment.

## Power and Ground Loop Analysis
Identify and explain concerns such as: decoupling capacitor placement and
values versus IC power pins; missing bulk or bypass capacitance; star-grounding
versus multi-point grounding conflicts; ground return path length for
high-current or high-speed loops; split or isolated ground schemes and their
stitching; power sequencing risks; and large physical loop areas implied by
how power and return nets are drawn.

## Common Mode Coupling Analysis
Identify and explain concerns such as: cable or connector shield grounding;
common-mode choke usage or absence on I/O and power lines; isolation barrier
crossings; unbalanced differential routing that could convert differential
noise to common-mode; and proximity of noisy switching nets to sensitive
analog or shield references.

## Prioritized Recommendations
Provide a numbered list of the highest-impact findings first.

For each recommendation, use this exact structure:

### 1. Short recommendation title
- **Issue:** What was observed or what is missing.
- **Risk:** What could happen if it remains unresolved.
- **Recommended fix:** The specific schematic-level action.
- **Evidence:** Reference the component, net, page, or visible design detail.
- **Confidence:** Confirmed / High probability / Potential risk / Missing information.

Keep each recommendation concise, technically specific, and easy to scan.

Formatting and evidence rules:
- Reference specific component designators, net names, or page numbers whenever
  you can see them in the image.
- If image quality or resolution prevents confirmation, say so explicitly.
- Never invent measurements, simulations, compliance claims, or test results.
- Separate engineering severity from AI confidence.
- Analyze only what can reasonably be established from the supplied images.
- Mark missing information clearly instead of guessing.
- Be direct and technical; this report is for a hardware engineer.
""".strip()


def build_gemini_contents(
    images: list[Image.Image],
    extra_notes: str,
) -> list:
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

def call_gemini(
    api_key: str,
    model: str,
    contents: list,
) -> str:
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
    """Split a Markdown report into sections while preserving content."""

    sections: dict[str, str] = {}

    parts = re.split(r"\n(?=##\s+)", report_text.strip())

    for part in parts:
        match = re.match(
            r"##\s+(.*?)\n(.*)",
            part.strip(),
            re.DOTALL,
        )

        if match:
            title = match.group(1).strip()
            body = match.group(2).strip()
            sections[title] = body

        elif part.strip():
            sections.setdefault("Other Notes", "")
            sections["Other Notes"] += part.strip() + "\n"

    return sections


# ---------------------------------------------------------------------------
# Streamlit UI styling
# ---------------------------------------------------------------------------

def configure_page():
    st.set_page_config(
        page_title="PCB/Schematic Reviewer",
        page_icon="🟩",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

        :root {
            --bg: #07110d;
            --surface: #0d1813;
            --surface-2: #111f18;
            --surface-3: #16271e;

            --primary: #65d46e;
            --primary-soft: #a8f0ad;
            --accent: #2fbf71;

            --text: #f4faf5;
            --muted: #c0cec3;

            --border: rgba(101, 212, 110, .22);
            --border-soft: rgba(255,255,255,.10);
        }

        html, body, [class*="css"] {
            font-family: Inter, sans-serif;
            color: var(--text);
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 50% -8%,
                    rgba(47,191,113,.12),
                    transparent 34%
                ),
                linear-gradient(180deg, #08120e 0%, #050a07 100%);
        }

        .block-container {
            max-width: 1320px;
            padding: 2.4rem 2rem 4.5rem;
        }

        /* =========================================================
           HERO
           ========================================================= */

        .pcb-board {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 2.35rem 2.5rem;
            margin-bottom: 1.35rem;
            background:
                linear-gradient(
                    135deg,
                    rgba(47,191,113,.10),
                    rgba(13,24,19,.94) 52%
                ),
                #0a140f;
            box-shadow: 0 18px 60px rgba(0,0,0,.30);
        }

        .pcb-board:before {
            content: "";
            position: absolute;
            inset: 0;
            opacity: .23;
            background:
                linear-gradient(
                    90deg,
                    transparent 49.7%,
                    rgba(101,212,110,.16) 50%,
                    transparent 50.3%
                ),
                linear-gradient(
                    transparent 49.7%,
                    rgba(101,212,110,.12) 50%,
                    transparent 50.3%
                );
            background-size: 46px 46px;
            pointer-events: none;
        }

        .eyebrow {
            position: relative;
            color: var(--primary);
            font-family: "JetBrains Mono", monospace;
            font-size: .72rem;
            font-weight: 600;
            letter-spacing: .13em;
            text-transform: uppercase;
        }

        .pcb-board h1 {
            position: relative;
            font-size: clamp(2.35rem, 4.4vw, 4rem);
            line-height: 1.04;
            letter-spacing: -.045em;
            font-weight: 800;
            margin: .65rem 0 .85rem;
            color: #f7fbf8;
        }

        .pcb-board h1 span {
            color: var(--primary);
        }

        .hero-copy {
            position: relative;
            max-width: 760px;
            color: #c8d7cc;
            font-size: .98rem;
            line-height: 1.7;
        }

        .trace-status {
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: .5rem;
            padding: .42rem .72rem;
            margin-top: 1rem;
            border: 1px solid rgba(101,212,110,.25);
            border-radius: 999px;
            background: rgba(101,212,110,.055);
            color: var(--primary-soft);
            font-family: "JetBrains Mono", monospace;
            font-size: .68rem;
            font-weight: 600;
            letter-spacing: .04em;
        }

        .led {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--primary);
            box-shadow: 0 0 10px rgba(101,212,110,.65);
        }

        /* =========================================================
           WORKFLOW
           ========================================================= */

        .workflow {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: .9rem;
            margin: 0 0 1.35rem;
        }

        .step {
            border: 1px solid var(--border-soft);
            border-radius: 15px;
            padding: .95rem 1.05rem;
            background: rgba(13,24,19,.78);
        }

        .step-num {
            display: inline-grid;
            place-items: center;
            width: 28px;
            height: 28px;
            margin-right: .55rem;
            border-radius: 50%;
            background: var(--primary);
            color: #07110d;
            font-size: .78rem;
            font-weight: 800;
        }

        .step b {
            color: #f0f6f1;
            font-size: .9rem;
        }

        .step small {
            display: block;
            margin-top: .35rem;
            color: var(--muted);
            font-size: .75rem;
            line-height: 1.5;
        }

        /* =========================================================
           FEATURE CARDS
           ========================================================= */

        .feature-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: .9rem;
            margin: 0 0 1.65rem;
        }

        .feature {
            min-height: 125px;
            border: 1px solid var(--border-soft);
            border-radius: 16px;
            padding: 1.15rem;
            background:
                linear-gradient(
                    145deg,
                    rgba(17,31,24,.92),
                    rgba(9,17,13,.94)
                );
            box-shadow: 0 8px 25px rgba(0,0,0,.12);
        }

        .feature .icon {
            font-size: 1.3rem;
        }

        .feature b {
            display: block;
            margin: .42rem 0 .28rem;
            color: var(--primary-soft);
            font-size: .88rem;
            font-weight: 700;
        }

        .feature span {
            color: var(--muted);
            font-size: .75rem;
            line-height: 1.55;
        }

        .section-label {
            color: var(--primary);
            font-family: "JetBrains Mono", monospace;
            font-size: .68rem;
            font-weight: 600;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin: 1.5rem 0 .7rem;
        }

        /* =========================================================
           UPLOAD / FORM
           ========================================================= */

        .upload-shell {
            border: 1px dashed rgba(101,212,110,.36);
            border-radius: 16px;
            padding: .65rem;
            background: rgba(101,212,110,.025);
        }

        [data-testid="stFileUploaderDropzone"] {
            background:
                linear-gradient(
                    135deg,
                    #173a28 0%,
                    #10261c 55%,
                    #0c1812 100%
                ) !important;
            border-radius: 13px !important;
            border: 1.5px solid rgba(101,212,110,.62) !important;
            box-shadow:
                inset 0 0 30px rgba(101,212,110,.045),
                0 8px 26px rgba(0,0,0,.18) !important;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #9be6a0 !important;
            box-shadow:
                inset 0 0 34px rgba(101,212,110,.08),
                0 10px 30px rgba(47,191,113,.12) !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #eef7f0 !important;
            font-size: .84rem !important;
            font-weight: 600 !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: #65d46e !important;
            color: #07110d !important;
            border: 1px solid #65d46e !important;
            border-radius: 9px !important;
            font-weight: 800 !important;
        }

        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #9be6a0 !important;
            border-color: #9be6a0 !important;
        }

        textarea,
        input {
            background: #26352c !important;
            color: #f4f8f5 !important;
            border: 1px solid rgba(101,212,110,.35) !important;
            border-radius: 10px !important;
            font-size: .86rem !important;
        }

        textarea:focus,
        input:focus {
            border-color: #8eea98 !important;
            box-shadow: 0 0 0 1px #8eea98 !important;
        }

        /* Prominent design-context field */
        div[data-testid="stTextArea"] label p {
            color: #f1f7f2 !important;
            font-size: .98rem !important;
            font-weight: 800 !important;
            letter-spacing: -.01em !important;
        }

        div[data-testid="stTextArea"] label p:before {
            content: "✦ ";
            color: var(--primary);
        }

        div[data-testid="stTextArea"] {
            padding: .15rem;
            border-radius: 13px;
        }

        div[data-testid="stTextArea"] > div {
            border-radius: 12px;
            box-shadow: 0 0 0 1px rgba(101,212,110,.05);
        }

        textarea::placeholder,
        input::placeholder {
            color: #cbd8ce !important;
            opacity: 1 !important;
        }

        /* =========================================================
           STATUS / METRICS
           ========================================================= */

        .status-card,
        .metric-card,
        .report-shell {
            border: 1px solid var(--border-soft);
            border-radius: 16px;
            background: #101d16 !important;
            color: #f4f8f5 !important;
        }

        .status-card {
            padding: 1.05rem 1.15rem;
        }

        .metric-card {
            padding: .9rem 1rem;
            min-height: 72px;
        }

        .metric-number {
            font-size: 1.35rem;
            line-height: 1.1;
            font-weight: 800;
            color: var(--primary);
        }

        .metric-label {
            margin-top: .28rem;
            color: var(--muted);
            font-size: .72rem;
        }

        /* =========================================================
           BUTTONS
           ========================================================= */

        .stButton > button,
        .stDownloadButton > button {
            background: var(--primary) !important;
            color: #07110d !important;
            border: 1px solid var(--primary) !important;
            border-radius: 10px !important;
            font-size: .84rem !important;
            font-weight: 800 !important;
            min-height: 2.75rem;
            box-shadow: 0 7px 22px rgba(47,191,113,.12);
            transition: all .18s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: #82df88 !important;
            border-color: #82df88 !important;
            color: #061009 !important;
            transform: translateY(-1px);
        }

        /* =========================================================
           REPORT TABS
           ========================================================= */

        .stTabs [data-baseweb="tab-list"] {
            gap: .45rem;
            border-bottom: 1px solid rgba(101, 212, 110, .16);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            padding: .65rem .9rem;
            background: #14251b;
            color: #d6e5d9 !important;
            font-size: .8rem;
            font-weight: 700;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: #1b3023;
            color: #ffffff !important;
        }

        .stTabs [aria-selected="true"] {
            background: #23452d !important;
            color: #a8f0ad !important;
            font-weight: 800 !important;
        }

        /* =========================================================
           REPORT CONTAINER
           ========================================================= */

        .report-shell {
            padding: 1.35rem 1.5rem;
            line-height: 1.75;
        }

        .report-title {
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -.025em;
            color: #ffffff;
            margin-bottom: .25rem;
        }

        .report-subtitle {
            color: #c0cec3;
            font-family: "JetBrains Mono", monospace;
            font-size: .66rem;
            letter-spacing: .04em;
        }

        .small-note {
            color: #c0cec3;
            font-size: .72rem;
            line-height: 1.55;
        }

        /* =========================================================
           REPORT MARKDOWN — HIGH CONTRAST
           ========================================================= */

        .report-shell [data-testid="stMarkdownContainer"] {
            color: #e8f1ea !important;
        }

        .report-shell p {
            color: #e8f1ea !important;
            font-size: .95rem !important;
            line-height: 1.75 !important;
            margin-bottom: .85rem !important;
        }

        .report-shell h1,
        .report-shell h2,
        .report-shell h3,
        .report-shell h4 {
            color: #ffffff !important;
            font-weight: 800 !important;
            line-height: 1.3 !important;
            margin-top: 1.35rem !important;
            margin-bottom: .65rem !important;
        }

        .report-shell h1:first-child,
        .report-shell h2:first-child,
        .report-shell h3:first-child {
            margin-top: 0 !important;
        }

        .report-shell h2 {
            color: #a8f0ad !important;
            font-size: 1.25rem !important;
            border-bottom: 1px solid rgba(101, 212, 110, .18);
            padding-bottom: .45rem;
        }

        .report-shell h3 {
            color: #ffffff !important;
            font-size: 1.05rem !important;
        }

        .report-shell ul,
        .report-shell ol {
            color: #e8f1ea !important;
            padding-left: 1.45rem !important;
            margin-bottom: 1rem !important;
        }

        .report-shell li {
            color: #e8f1ea !important;
            font-size: .94rem !important;
            line-height: 1.7 !important;
            margin-bottom: .45rem !important;
        }

        .report-shell li::marker {
            color: #65d46e !important;
        }

        .report-shell strong,
        .report-shell b {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        .report-shell em,
        .report-shell i {
            color: #c8e2cd !important;
        }

        .report-shell code {
            color: #b8f0bd !important;
            background: #1b3023 !important;
            border: 1px solid rgba(101, 212, 110, .18);
            border-radius: 5px;
            padding: .12rem .35rem;
            font-size: .86em;
        }

        .report-shell pre {
            background: #0a140f !important;
            border: 1px solid rgba(101, 212, 110, .18) !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            overflow-x: auto;
        }

        .report-shell pre code {
            background: transparent !important;
            border: none !important;
            color: #d8f5dc !important;
        }

        .report-shell a {
            color: #8eea98 !important;
            text-decoration: underline;
            text-underline-offset: 3px;
        }

        .report-shell blockquote {
            border-left: 3px solid #65d46e !important;
            background: #16271e !important;
            color: #d5e8d8 !important;
            padding: .7rem 1rem !important;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }

        .report-shell hr {
            border: none !important;
            border-top: 1px solid rgba(101, 212, 110, .2) !important;
            margin: 1.3rem 0 !important;
        }

        .report-shell table {
            width: 100% !important;
            border-collapse: collapse !important;
            margin: 1rem 0 !important;
            color: #e8f1ea !important;
        }

        .report-shell th {
            background: #1b3023 !important;
            color: #ffffff !important;
            font-weight: 800 !important;
            text-align: left !important;
            padding: .75rem !important;
            border: 1px solid rgba(101, 212, 110, .2) !important;
        }

        .report-shell td {
            color: #e8f1ea !important;
            padding: .7rem !important;
            border: 1px solid rgba(255, 255, 255, .10) !important;
            background: #101d16 !important;
        }

        .report-shell tr:nth-child(even) td {
            background: #14251b !important;
        }

        /* Streamlit alerts and status text */
        [data-testid="stAlert"] {
            color: #eef7f0 !important;
        }

        [data-testid="stStatusWidget"] {
            color: #eef7f0 !important;
        }

        /* =========================================================
           RESPONSIVE DESIGN
           ========================================================= */

        @media (max-width: 800px) {
            .feature-row,
            .workflow {
                grid-template-columns: 1fr;
            }

            .pcb-board {
                padding: 1.55rem;
            }

            .block-container {
                padding: 1.15rem 1rem 3rem;
            }

            .pcb-board h1 {
                font-size: 2.35rem;
            }

            .report-shell {
                padding: 1.1rem;
            }

            .report-shell p,
            .report-shell li {
                font-size: .9rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric(number: str, label: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{number}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Report export helpers
# ---------------------------------------------------------------------------

def make_pdf(
    report_text: str,
    filename: str = "pcb_schematic_review.pdf",
) -> bytes:
    """Create a simple PDF report from the generated Markdown text."""

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle,
    )
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
    )
    from reportlab.lib.units import mm

    buf = BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "PCBTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#111111"),
        spaceAfter=10,
    )

    body = ParagraphStyle(
        "PCBBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#222222"),
        spaceAfter=7,
    )

    heading = ParagraphStyle(
        "PCBHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#111111"),
        spaceBefore=10,
        spaceAfter=6,
    )

    story = [
        Paragraph(
            "PCB/Schematic Reviewer — Engineering Report",
            title,
        ),
        Paragraph(
            "AI-assisted engineering review",
            body,
        ),
        Spacer(1, 5),
    ]

    for raw in report_text.splitlines():
        line = raw.strip()

        if not line:
            story.append(Spacer(1, 4))

        elif line.startswith("#"):
            text = re.sub(r"^#+\s*", "", line)
            story.append(Paragraph(escape(text), heading))

        elif line.startswith(("- ", "* ")):
            story.append(
                Paragraph("• " + escape(line[2:]), body)
            )

        else:
            story.append(Paragraph(escape(line), body))

    doc.build(story)
    return buf.getvalue()


def make_docx(report_text: str) -> bytes:
    """Create a Word report from the generated Markdown text."""

    from docx import Document
    from docx.shared import Pt

    doc = Document()

    doc.add_heading(
        "PCB/Schematic Reviewer — Engineering Report",
        level=0,
    )

    p = doc.add_paragraph("AI-assisted engineering review")
    p.runs[0].bold = True

    for raw in report_text.splitlines():
        line = raw.strip()

        if not line:
            continue

        if line.startswith("#"):
            level = min(
                len(line) - len(line.lstrip("#")),
                3,
            )
            doc.add_heading(
                re.sub(r"^#+\s*", "", line),
                level=level,
            )

        elif line.startswith(("- ", "* ")):
            doc.add_paragraph(
                line[2:],
                style="List Bullet",
            )

        else:
            doc.add_paragraph(line)

    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Arial"
            run.font.size = Pt(10)

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def render_empty_state():
    st.markdown(
        """
        <div class="status-card" style="text-align:center; padding:2.4rem 1.5rem;">
            <div style="font-size:2.5rem;">🟩</div>
            <h3 style="margin:.5rem 0; color:#ffffff;">
                Your engineering review starts here
            </h3>
            <div style="color:#c0cec3; max-width:650px; margin:auto;">
                Add a schematic PDF, optionally tell us about the design,
                and let the reviewer surface the highest-value engineering risks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    configure_page()

    settings = {
        "api_key": st.secrets.get("GEMINI_API_KEY", None),
        "model": DEFAULT_MODEL,
        "zoom": 2.0,
    }

    st.markdown(
        """
        <section class="pcb-board">
            <div class="eyebrow">PCB / SCHEMATIC REVIEWER</div>
            <h1>
                Review your design.<br>
                <span>Catch risks earlier.</span>
            </h1>
            <div class="hero-copy">
                AI-assisted engineering analysis for signal integrity,
                power & ground, and EMI/common-mode coupling —
                presented as practical, prioritized findings.
            </div>
            <div class="trace-status">
                <span class="led"></span>
                ANALYSIS CONSOLE ONLINE
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="workflow">
            <div class="step">
                <span class="step-num">1</span>
                <b>Add schematic</b>
                <small>
                    Upload your PDF and optionally add design context.
                </small>
            </div>
            <div class="step">
                <span class="step-num">2</span>
                <b>Analyze PCB</b>
                <small>
                    Run the AI review and inspect prioritized engineering findings.
                </small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-row">
            <div class="feature">
                <div class="icon">⚡</div>
                <b>Signal Integrity</b>
                <span>
                    Termination, reflections, interfaces and high-speed risk areas.
                </span>
            </div>
            <div class="feature">
                <div class="icon">🔋</div>
                <b>Power & Ground</b>
                <span>
                    Decoupling, return paths, loops and power integrity concerns.
                </span>
            </div>
            <div class="feature">
                <div class="icon">📡</div>
                <b>EMI & Coupling</b>
                <span>
                    Common-mode paths, filtering, shields and coupling risks.
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">01 · Add schematic</div>',
        unsafe_allow_html=True,
    )

    upload_col, context_col = st.columns(
        [1.3, 1],
        gap="large",
    )

    with upload_col:
        st.markdown(
            '<div class="upload-shell">',
            unsafe_allow_html=True,
        )

        uploaded_pdf = st.file_uploader(
            "📄 Drag & drop your schematic PDF",
            type=["pdf"],
            help="Multi-page schematic PDFs are supported.",
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        if uploaded_pdf:
            st.success(
                f"✓ {uploaded_pdf.name} loaded • "
                f"{uploaded_pdf.size / 1024:.0f} KB",
                icon="🟩",
            )

    with context_col:
        st.markdown(
            """
            <div style="font-size:.98rem;font-weight:800;color:#f1f7f2;margin:0 0 .35rem;">
                ✦ Design context
                <span style="color:#c0cec3;font-size:.78rem;font-weight:600;">
                    (optional)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        extra_notes = st.text_area(
            "Design context (optional)",
            placeholder=(
                "Example: 4-layer board • USB 3.x • buck regulator on page 2 "
                "• sensitive ADC on page 4"
            ),
            height=132,
            label_visibility="collapsed",
        )

    if uploaded_pdf:
        st.markdown(
            '<div class="section-label">02 · Analyze PCB</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            render_metric("✓", "Schematic ready")

        with c2:
            render_metric(
                f"{settings['zoom']:.1f}×",
                "Render quality",
            )

        with c3:
            render_metric("AI", "Engineering review")

        st.write("")

        run_clicked = st.button(
            "⚡ Analyze My PCB/Schematic",
            type="primary",
            use_container_width=True,
        )

    else:
        run_clicked = False
        render_empty_state()

    if run_clicked:
        if not settings["api_key"]:
            st.error(
                "GEMINI_API_KEY is missing from Streamlit Secrets. "
                "Add it before running the review."
            )
            return

        pdf_bytes = uploaded_pdf.getvalue()

        with st.status(
            "Preparing schematic…",
            expanded=True,
        ) as status:
            st.write("Rendering schematic pages.")

            images = render_pdf_pages_to_images(
                pdf_bytes,
                zoom=settings["zoom"],
            )

            st.write(f"✓ {len(images)} page(s) prepared")

            status.update(
                label="Schematic ready",
                state="complete",
                expanded=False,
            )

        with st.expander(
            f"👁 Preview submitted schematic • {len(images)} page(s)",
            expanded=False,
        ):
            cols = st.columns(min(len(images), 4) or 1)

            for i, img in enumerate(images):
                cols[i % len(cols)].image(
                    img,
                    caption=f"Page {i + 1}",
                    use_container_width=True,
                )

        try:
            with st.status(
                "🤖 Analyzing your PCB/schematic…",
                expanded=True,
            ) as status:
                st.write(
                    "Checking signal integrity, power/ground, "
                    "and EMI & coupling."
                )

                contents = build_gemini_contents(
                    images,
                    extra_notes,
                )

                report_text = call_gemini(
                    settings["api_key"],
                    settings["model"],
                    contents,
                )

                status.update(
                    label="Analysis complete",
                    state="complete",
                    expanded=False,
                )

        except RuntimeError as exc:
            st.error(f"Analysis failed: {exc}")
            return

        st.session_state["last_report"] = report_text

    if "last_report" in st.session_state:
        st.markdown(
            '<div class="section-label">Engineering report</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="report-shell">
                <div class="report-title">
                    PCB/Schematic Review Report
                </div>
                <div class="report-subtitle">
                    AI-ASSISTED • PRIORITIZED ENGINEERING FINDINGS
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        sections = split_report_into_sections(
            st.session_state["last_report"]
        )

        tab_titles = [
            title
            for title in REPORT_SECTIONS
            if title in sections
        ]

        tab_titles += [
            title
            for title in sections
            if title not in REPORT_SECTIONS
        ]

        tabs = st.tabs(
            [f"  {title}" for title in tab_titles]
        )

        for tab, title in zip(tabs, tab_titles):
            with tab:
                st.markdown(
                    '<div class="report-shell">',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    sections[title],
                    unsafe_allow_html=False,
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

        st.write("")

        pdf_bytes = make_pdf(
            st.session_state["last_report"]
        )

        docx_bytes = make_docx(
            st.session_state["last_report"]
        )

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "⬇️ Download PDF report",
                data=pdf_bytes,
                file_name="pcb_schematic_review.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with d2:
            st.download_button(
                "⬇️ Download Word report",
                data=docx_bytes,
                file_name="pcb_schematic_review.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
