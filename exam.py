"""
AI Study Assistant
Single-file Streamlit app implementing:
- Upload PDF or paste text
- Extract text from PDF (pdfplumber or PyPDF2)
- Use OpenAI API to generate summary, flashcards, Q&A, and mind-map (Graphviz DOT)
- Download summary (txt) and flashcards (JSON)
- Sidebar controls for model selection and counts

Usage:
1. Install dependencies: pip install -r requirements.txt
   Example requirements.txt contents (put in same repo):
     streamlit
     openai
     pdfplumber
     PyPDF2
     graphviz

2. Set your OpenAI API key in environment:
   export OPENAI_API_KEY="sk-..."

3. Run locally:
   streamlit run streamlit_app.py

Deployment guidance (Streamlit Community Cloud / Render / Railway):
- Push this file and a requirements.txt to a public GitHub repo
- For Streamlit Cloud: connect repo and deploy. Streamlit will give you a public URL.
- For Render/Railway: follow their deployment steps; they will expose a public URL.

NOTE: This app *does not* itself automatically publish a URL. When you deploy to Streamlit Cloud or similar, the platform gives the public URL.

"""

import os
import json
import io
import time
from typing import Optional

import streamlit as st

# PDF extraction libraries: try pdfplumber first, fall back to PyPDF2
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

try:
    from PyPDF2 import PdfReader
    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False

# OpenAI SDK - using openai package
try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# Streamlit app layout
st.set_page_config(page_title="AI Study Assistant", page_icon=":mortar_board:", layout="wide")

st.title("AI Study Assistant")
st.markdown("A lightweight Streamlit app to summarize study material, create flashcards, short Q&A, and a mind-map using the OpenAI API.")

# Sidebar controls
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", help="Alternatively set the OPENAI_API_KEY environment variable.")
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

if not os.getenv("OPENAI_API_KEY"):
    st.sidebar.error("OpenAI API key not found. Enter it above or set OPENAI_API_KEY environment variable.")

MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-std"]
model = st.sidebar.selectbox("Model", MODEL_OPTIONS, index=0)

num_bullets = st.sidebar.slider("Number of summary bullets", min_value=3, max_value=10, value=5)
num_flashcards = st.sidebar.slider("Number of flashcards", min_value=3, max_value=30, value=10)
num_short_qa = st.sidebar.slider("Number of short Q&A pairs", min_value=3, max_value=20, value=8)

use_pdfplumber = st.sidebar.checkbox("Prefer pdfplumber for PDF extraction (if installed)", value=True)
max_tokens = st.sidebar.slider("Max tokens per API call", min_value=256, max_value=8192, value=1500)

st.sidebar.markdown("---")
st.sidebar.markdown("**Deployment**: Push this file + requirements.txt to GitHub and deploy on Streamlit Community Cloud. The platform will provide a public URL you can share.")

# Input area: file upload or paste
st.subheader("Input")
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], help="Upload a PDF file to extract text")
    paste_text = st.text_area("Or paste text here (overrides uploaded file)", height=200)

with col2:
    st.info("Paste text to use it directly. If both are provided, pasted text takes precedence. For PDFs, extraction attempts to preserve reading order.")

@st.cache_data(show_spinner=False)
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text_pages = []
    if _HAS_PDFPLUMBER and use_pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    text_pages.append(p.extract_text() or "")
            return "\n\n".join(text_pages).strip()
        except Exception:
            pass

    if _HAS_PYPDF2:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                try:
                    text_pages.append(page.extract_text() or "")
                except Exception:
                    text_pages.append("")
            return "\n\n".join(text_pages).strip()
        except Exception:
            pass

    # Fallback: return empty string
    return ""

# Aggregate input text
input_text = ""
if paste_text and paste_text.strip():
    input_text = paste_text.strip()
elif uploaded_file is not None:
    try:
        pdf_bytes = uploaded_file.read()
        input_text = extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")

if not input_text:
    st.warning("No input provided yet. Paste text or upload a PDF to get started.")
    st.stop()

st.subheader("Preview of extracted / pasted text")
with st.expander("Show text (first 5000 chars)", expanded=False):
    st.text(input_text[:5000] + ("\n\n..." if len(input_text) > 5000 else ""))

# OpenAI helper
def call_openai_chat(system_prompt: str, user_prompt: str, model_name: str = "gpt-4o-mini", temperature: float = 0.1, max_tokens_local: int = 1500):
    if not _HAS_OPENAI:
        raise RuntimeError("openai package not installed in the environment.")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    # Use ChatCompletion for broad compatibility
    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens_local,
        )
        out = resp.choices[0].message.content
        return out
    except Exception as e:
        # Return readable error
        return f"[OpenAI call failed] {e}"

# Generate tasks
st.subheader("Generate Outputs")
if st.button("Generate summary, flashcards, Q&A & mind-map"):
    with st.spinner("Calling OpenAI and generating outputs..."):
        # Prepare context-limited text: trim to model token budget (very simple heuristic: characters)
        context_text = input_text
        if len(context_text) > 60000:
            context_text = context_text[:60000] + "\n\n[TRUNCATED]"

        # 1) Summary prompt
        system_prompt = "You are an expert study assistant. Produce concise outputs as instructed."
        user_prompt_summary = (
            f"Read the following content and produce a concise summary in {num_bullets} bullet points. Keep each bullet short (one sentence).\n\nCONTENT:\n{context_text}"
        )
        summary_raw = call_openai_chat(system_prompt, user_prompt_summary, model_name=model, temperature=0.2, max_tokens_local= max(200, num_bullets*40))

        # 2) Flashcards prompt (JSON array of {question, answer})
        user_prompt_flashcards = (
            f"From the content below, create exactly {num_flashcards} educational flashcards in JSON array format. "
            "Each flashcard must be an object with 'question' and 'answer' fields. Keep questions concise and answers 1-3 sentences. "
            f"CONTENT:\n{context_text}"
        )
        flashcards_raw = call_openai_chat(system_prompt, user_prompt_flashcards, model_name=model, temperature=0.3, max_tokens_local= max(400, num_flashcards*80))

        # 3) Short Q&A pairs
        user_prompt_qa = (
            f"Create {num_short_qa} short question-answer pairs useful for quick revision. Output as plain text with 'Q: ...' and 'A: ...' lines. "
            f"CONTENT:\n{context_text}"
        )
        short_qa_raw = call_openai_chat(system_prompt, user_prompt_qa, model_name=model, temperature=0.3, max_tokens_local= max(300, num_short_qa*60))

        # 4) Mind-map DOT (Graphviz) - ask model to produce DOT only
        user_prompt_mindmap = (
            "Create a simple mind-map for the content using Graphviz DOT format. "
            "Focus on main topics (as root nodes) and 2-4 child nodes each. Output ONLY the DOT code starting with 'digraph' or 'graph'.\n\n"
            f"CONTENT:\n{context_text}"
        )
        mindmap_raw = call_openai_chat(system_prompt, user_prompt_mindmap, model_name=model, temperature=0.4, max_tokens_local=800)

    # Display outputs in sections
    st.success("Generation complete — results below")

    # SUMMARY
    st.header("Summary")
    st.markdown(summary_raw)
    # Provide download for summary txt
    summary_file_bytes = summary_raw.encode("utf-8")
    st.download_button("Download summary (TXT)", data=summary_file_bytes, file_name="summary.txt", mime="text/plain")

    # FLASHCARDS
    st.header("Flashcards (JSON)")
    # Try to parse flashcards_raw as JSON; if fails, show raw and offer to download raw text
    parsed_flashcards = None
    try:
        parsed_flashcards = json.loads(flashcards_raw)
    except Exception:
        # Attempt simple extraction: if the model returned a JSON block inside text, try to extract {...}
        try:
            start = flashcards_raw.index("[")
            end = flashcards_raw.rindex("]") + 1
            parsed_flashcards = json.loads(flashcards_raw[start:end])
        except Exception:
            parsed_flashcards = None

    if parsed_flashcards is not None:
        st.write(parsed_flashcards)
        st.download_button("Download flashcards (JSON)", data=json.dumps(parsed_flashcards, indent=2).encode("utf-8"), file_name="flashcards.json", mime="application/json")
    else:
        st.warning("Could not parse flashcards as JSON automatically. Here's the raw output — you can copy it and save as .json if desired.")
        st.text_area("Flashcards (raw)", flashcards_raw, height=300)
        st.download_button("Download flashcards (raw).txt", data=flashcards_raw.encode("utf-8"), file_name="flashcards_raw.txt", mime="text/plain")

    # SHORT QA
    st.header("Short Q&A")
    st.text_area("Short Q&A pairs", short_qa_raw, height=250)
    st.download_button("Download Q&A (TXT)", data=short_qa_raw.encode("utf-8"), file_name="short_qa.txt", mime="text/plain")

    # MIND-MAP
    st.header("Mind-map (Graphviz DOT)")
    st.subheader("DOT code")
    st.code(mindmap_raw, language="dot")
    # Try to render via Streamlit's graphviz_chart
    try:
        st.graphviz_chart(mindmap_raw)
    except Exception as e:
        st.error(f"Could not render Graphviz chart: {e}")

    # Show footer / usage tips
    st.markdown("---")
    st.markdown("**Tips:** If any output looks wrong, try increasing temperature or trimming very long input text. For large PDFs, split by chapter and process separately.")

# End of app

# Small helper: when the file is deployed, the hosting platform will provide a public URL; include a small instructions block to explain how to obtain it
st.sidebar.markdown("---")
st.sidebar.header("Get a public URL (deployment)")
st.sidebar.markdown(
    "1. Create a GitHub repo with this file and a requirements.txt.\n" 
    "2. Go to Streamlit Community Cloud (https://streamlit.io/cloud), connect the repo, and deploy.\n"
    "3. After deployment, Streamlit will provide a public URL you can share and open in Google.\n\n"
    "Alternatively, use Render or Railway; they also provide public URLs after deployment.")

# End of file
