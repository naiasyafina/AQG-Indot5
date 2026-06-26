"""
AQG — Automatic Question Generator Bahasa Indonesia
Streamlit App: Upload PDF/PPTX/DOCX → Preprocessing → Generate Soal → Quiz

Model: onalla/AQGINDOT5-QTYPE (IndoT5 fine-tuned pada IDK-MRC)
Format input model:
    generate question: question_type: {question_type} answer: {answer} context: {context}

Kelompok:
    Naia Syafina H.  — A11.2024.15554
    Onalla Aldeanuva — A11.2024.15952
"""

import io
import json
import os
import random
import re
import warnings
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import torch

warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURASI GLOBAL
# =============================================================================

HF_MODEL_REPO      = "onalla/AQGINDOT5-QTYPE"
MAX_INPUT_LEN      = 1024
MAX_TARGET_LEN     = 64
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_QWEN_MODEL = "qwen/qwen3-32b"

GEN_CONFIG = dict(
    max_length           = MAX_TARGET_LEN,
    min_length           = 5,
    num_beams            = 4,
    repetition_penalty   = 1.2,
    length_penalty       = 1.0,
    early_stopping       = True,
    no_repeat_ngram_size = 3,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AQG — Question Generator",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GLOBAL CSS
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --blue-900:  #1E3A5F;
    --blue-700:  #2563A8;
    --blue-500:  #4A90D9;
    --blue-300:  #93C4EE;
    --blue-100:  #DBEAFE;
    --blue-50:   #EFF6FF;
    --bg-main:   #F4F7FF;
    --bg-card:   #FFFFFF;
    --bg-muted:  #E8F0FD;
    --text-dark: #1A2B45;
    --text-mid:  #3D5A80;
    --text-soft: #6B87A8;
    --border:    #C8DCF5;
    --border-s:  #DDEAF9;
    --green-bg:  #E8F7F2;
    --green-txt: #0F7B55;
    --red-bg:    #FEF2F2;
    --red-txt:   #B91C1C;
}

html, body, [class*="css"], .stMarkdown, p, span, label, div {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
#MainMenu, footer { visibility: hidden; }

.main .block-container {
    background: var(--bg-main) !important;
    padding-top: 2rem;
    padding-bottom: 5rem;
    max-width: 740px;
}

/* Sidebar biru gelap */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1E3A5F 0%, #2563A8 100%) !important;
}
[data-testid="stSidebar"] * { color: #E8F4FF !important; }
[data-testid="stSidebar"] hr { border-color: rgba(147,196,238,0.25) !important; }
[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [data-testid="stCheckbox"] span { color: #E8F4FF !important; }
[data-testid="stSidebar"] [data-testid="stTextInput"] label { color: rgba(147,196,238,0.85) !important; }
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.12) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(147,196,238,0.3) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
    color: rgba(147,196,238,0.5) !important;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, var(--blue-900) 0%, var(--blue-700) 100%);
    border-radius: 18px; padding: 1.8rem 1.6rem 1.4rem;
    margin-bottom: 1.8rem; position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:180px; height:180px; background:rgba(255,255,255,0.04); border-radius:50%;
}
.hero-eyebrow { font-size:0.62rem; letter-spacing:0.22em; text-transform:uppercase; color:var(--blue-300); font-weight:600; margin-bottom:0.6rem; }
.hero-title { font-family:'Lora',serif; font-size:1.55rem; line-height:1.2; color:#FFF; font-weight:600; margin:0 0 0.4rem; }
.hero-title em { font-style:italic; color:#93C4EE; font-weight:400; }
.hero-sub { font-size:0.78rem; color:rgba(147,196,238,0.85); font-weight:300; margin-top:0.4rem; }

/* Section label */
.sec-label {
    font-size:0.64rem; letter-spacing:0.16em; text-transform:uppercase;
    color:var(--text-soft); border-left:3px solid var(--blue-500);
    padding-left:0.7rem; margin-bottom:0.9rem; font-weight:600;
}

/* Card */
.card {
    background: var(--bg-card); border: 1px solid var(--border-s);
    border-radius: 14px; padding: 1.5rem 1.7rem; margin-bottom: 1rem; color: var(--text-dark);
}
.card * { color: var(--text-dark) !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important; border: 2px dashed var(--border) !important; border-radius: 14px !important;
}
[data-testid="stFileUploader"] * { color: var(--text-dark) !important; }
[data-testid="stFileUploader"]:hover { border-color: var(--blue-500) !important; }

/* Slider */
.stSlider label, .stSlider span, .stSlider p { color: var(--text-dark) !important; }

/* Buttons */
.stButton > button {
    background: var(--blue-700) !important; color: #FFF !important;
    border: none !important; border-radius: 10px !important;
    font-size: 0.87rem !important; font-weight: 600 !important;
    padding: 0.65rem 1.8rem !important; transition: background 0.15s, transform 0.1s !important;
}
.stButton > button:hover { background: var(--blue-900) !important; transform: translateY(-1px) !important; }

/* Radio button — index=None, TIDAK pakai placeholder kosong */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div { gap: 8px !important; flex-direction: column !important; }
div[data-testid="stRadio"] > div > label {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.82rem 1.1rem !important;
    cursor: pointer !important;
    width: 100% !important; min-height: 50px !important;
    display: flex !important; align-items: center !important;
    box-sizing: border-box !important;
    transition: border-color 0.15s, background 0.15s !important;
}
div[data-testid="stRadio"] > div > label,
div[data-testid="stRadio"] > div > label *,
div[data-testid="stRadio"] > div > label span,
div[data-testid="stRadio"] > div > label p {
    color: var(--text-dark) !important; font-size: 0.88rem !important;
}
div[data-testid="stRadio"] > div > label:hover {
    border-color: var(--blue-500) !important; background: var(--blue-50) !important;
}
/* State checked */
div[data-testid="stRadio"] > div > label[data-checked="true"],
div[data-testid="stRadio"] > div > label:has(input:checked) {
    border-color: var(--blue-700) !important;
    background: var(--blue-50) !important;
}
div[data-testid="stRadio"] > div > label[data-checked="true"] span,
div[data-testid="stRadio"] > div > label[data-checked="true"] p {
    color: var(--blue-700) !important; font-weight: 600 !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important; border: 1px solid var(--border-s) !important; border-radius: 12px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] > div,
[data-testid="stExpander"] > div * { color: var(--text-dark) !important; }
[data-testid="stExpander"] pre,
[data-testid="stExpander"] code {
    background: #1A2B45 !important; color: #E8F4FF !important;
    border-radius: 8px !important; font-size: 0.78rem !important;
}

/* Progress bar quiz */
.prog-wrap { background: var(--blue-100); border-radius: 99px; height: 5px; margin-bottom: 1.5rem; overflow: hidden; }
.prog-fill { height: 100%; background: linear-gradient(90deg, var(--blue-700), var(--blue-500)); border-radius: 99px; transition: width 0.4s ease; }
.quiz-meta { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.7rem; }
.quiz-meta-label { font-size:0.62rem; letter-spacing:0.16em; text-transform:uppercase; color:var(--text-soft); font-weight:600; }
.quiz-meta-counter { font-family:'Lora',serif; font-size:1rem; color:var(--blue-700); font-weight:600; }
.quiz-q-label { font-size:0.6rem; letter-spacing:0.16em; text-transform:uppercase; color:var(--blue-500); font-weight:600; margin-bottom:0.4rem; }
.quiz-q-text { font-family:'Lora',serif; font-size:1.12rem; color:var(--text-dark); line-height:1.55; margin-bottom:1.3rem; }
.qtype-badge {
    display:inline-block; background:var(--bg-muted); color:var(--text-mid);
    border:1px solid var(--border); border-radius:99px;
    font-size:0.62rem; font-weight:600; padding:0.18rem 0.75rem;
    margin-bottom:1rem; letter-spacing:0.06em; text-transform:uppercase;
}

/* Result cards */
.result-card { background:var(--bg-card); border:1px solid var(--border-s); border-radius:12px; padding:1rem 1.3rem; margin-bottom:0.7rem; border-left:4px solid var(--border); }
.result-card.correct { border-left-color:#16A34A; background:var(--green-bg); }
.result-card.wrong   { border-left-color:#DC2626; background:var(--red-bg); }
.result-card * { color: var(--text-dark) !important; }
.result-card.correct .rc-label { color: var(--green-txt) !important; }
.result-card.wrong .rc-label   { color: var(--red-txt) !important; }

/* Score */
.score-wrap { text-align:center; padding:1.8rem 0 1.2rem; }
.score-big  { font-family:'Lora',serif; font-size:4rem; font-weight:600; color:var(--blue-700); line-height:1; }
.score-label { font-size:0.66rem; letter-spacing:0.16em; text-transform:uppercase; color:var(--text-soft); margin-top:0.3rem; font-weight:600; }
.score-grade { display:inline-block; background:var(--blue-50); color:var(--blue-700); border:1.5px solid var(--blue-100); border-radius:99px; font-size:0.72rem; font-weight:700; padding:0.25rem 1.1rem; margin-top:0.7rem; letter-spacing:0.08em; text-transform:uppercase; }

/* Debug */
.stTextArea textarea { color: var(--text-dark) !important; background: var(--bg-card) !important; }
pre, code { color: #E8F4FF !important; background: #1A2B45 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; }

/* Qwen status panel */
.qwen-panel {
    background: var(--bg-card); border: 1px solid var(--border-s);
    border-radius: 12px; padding: 1rem 1.3rem; margin-bottom: 1rem;
}
.qwen-panel * { color: var(--text-dark) !important; }
.qwen-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem; font-size:0.8rem; }
.qwen-key { color: var(--text-soft) !important; }
.qwen-val { font-weight:600; color: var(--text-dark) !important; }
.qwen-ok  { color: #0F7B55 !important; }
.qwen-err { color: #B91C1C !important; }
.qwen-warn{ color: #B45309 !important; }

@keyframes popIn {
    from { opacity:0; transform:scale(0.85) translateY(20px); }
    to   { opacity:1; transform:scale(1) translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0.2rem 0.4rem;">
        <div style="font-size:0.58rem;letter-spacing:0.22em;text-transform:uppercase;color:#93C4EE;font-weight:600;margin-bottom:0.4rem;">✦ NLP Project · 2026</div>
        <div style="font-family:'Georgia',serif;font-size:1.45rem;font-weight:700;color:#FFF;line-height:1.2;margin-bottom:0.25rem;">Automatic<br>Question<br>Generator</div>
        <div style="font-size:0.72rem;color:rgba(147,196,238,0.75);margin-top:0.3rem;line-height:1.5;">Fine-tuned IndoT5 untuk pembangkitan soal otomatis dari materi kuliah Bahasa Indonesia.</div>
    </div><hr>
    <div style="font-size:0.58rem;letter-spacing:0.18em;text-transform:uppercase;color:#93C4EE;font-weight:600;margin-bottom:0.7rem;">👥 Anggota Tim</div>
    """, unsafe_allow_html=True)

    for name, nim in [("Naia Syafina H.", "A11.2024.15554"), ("Onalla Aldeanuva", "A11.2024.15952")]:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;margin-bottom:0.65rem;background:rgba(255,255,255,0.08);border-radius:9px;padding:0.55rem 0.75rem;">
            <span style="font-size:0.82rem;font-weight:600;color:#FFF;">{name}</span>
            <span style="font-size:0.68rem;color:rgba(147,196,238,0.7);margin-top:1px;">{nim}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<hr>
    <div style="font-size:0.58rem;letter-spacing:0.18em;text-transform:uppercase;color:#93C4EE;font-weight:600;margin-bottom:0.7rem;">🤖 Model</div>""",
    unsafe_allow_html=True)

    for label, val in [("Model","IndoT5-QTYPE"),("Dataset","IDK-MRC"),("Device",DEVICE.upper()),("Framework","HuggingFace")]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
            <span style="font-size:0.7rem;color:rgba(147,196,238,0.7);">{label}</span>
            <span style="font-size:0.7rem;font-weight:600;color:#FFF;background:rgba(255,255,255,0.1);border-radius:5px;padding:0.12rem 0.45rem;">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.58rem;letter-spacing:0.18em;text-transform:uppercase;color:#93C4EE;font-weight:600;margin-bottom:0.5rem;">⚙️ Pengaturan Qwen</div>""",
    unsafe_allow_html=True)

    use_qwen = st.checkbox(
        "Aktifkan Qwen (validator & distractor)",
        value=False,
        help="Qwen via Groq dipakai sebagai validator/rewrite pertanyaan dan pembuat distractor. Jawaban benar tetap dari dokumen.",
    )
    qwen_model_name = st.text_input(
        "Model Qwen via Groq",
        value=DEFAULT_QWEN_MODEL,
        help="Contoh: qwen/qwen3-27b, qwen/qwen3-7b",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    show_debug = st.checkbox("Tampilkan review & download debug", value=False)

    st.markdown("""<div style="margin-top:1.5rem;font-size:0.62rem;color:rgba(147,196,238,0.45);text-align:center;">© 2026 · Kelompok AQG</div>""",
    unsafe_allow_html=True)

# =============================================================================
# HERO
# =============================================================================

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">✦ Automatic Question Generation · 2026</div>
    <h1 class="hero-title">Dari teks kuliah<br>ke pertanyaan <em>bermakna</em></h1>
    <div class="hero-sub">Upload PDF, PPTX, atau DOCX — soal pilihan ganda dihasilkan otomatis.</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# IMPORT PREPROCESS
# =============================================================================

try:
    from preprocess import (
        full_preprocess_pipeline as _preprocess_pipeline,
        pick_answer,
        remove_repeated_headers_footers,
        is_noise_unit,
        clean_option_text,
        is_clean_distractor,
        heuristic_validate_question,
        generate_heuristic_distractors,
        infer_question_type,
        get_focused_context,
        _TOC_LINE_RE,
        _COLOPHON_RE,
        _PUBLICATION_META_RE,
        score_content_chunk,
    )
    PREPROCESS_OK = True
except ImportError as _ie:
    PREPROCESS_OK = False
    st.error(f"❌ Gagal import preprocess.py: {_ie}")
    st.stop()


def full_preprocess_pipeline(raw_text: str) -> Dict[str, Any]:
    return _preprocess_pipeline(raw_text)


# =============================================================================
# GROQ — API KEY & PACKAGE CHECK
# =============================================================================

def get_groq_api_key() -> str:
    """
    Baca GROQ_API_KEY dari st.secrets atau environment variable.
    Tidak pernah menampilkan key di UI atau debug.
    """
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def check_groq_availability() -> Dict[str, Any]:
    """
    Cek ketersediaan package groq dan API key.
    Return dict status yang jujur, tanpa menyertakan key.

    {
        "package_available": bool,
        "api_key_found": bool,
        "package_error": str,  # jika import gagal
        "key_error": str,      # jika key tidak ada
    }
    """
    status: Dict[str, Any] = {
        "package_available": False,
        "api_key_found": False,
        "package_error": "",
        "key_error": "",
    }
    try:
        import groq  # noqa: F401
        status["package_available"] = True
    except ImportError:
        status["package_error"] = "Package 'groq' belum terinstall. Jalankan: pip install groq"

    api_key = get_groq_api_key()
    if api_key:
        status["api_key_found"] = True
    else:
        status["key_error"] = (
            "GROQ_API_KEY tidak ditemukan. "
            "Simpan di .streamlit/secrets.toml atau environment variable."
        )

    return status


def get_groq_client():
    """
    Buat Groq client. Return (client, error_msg).
    Tidak menyimpan key di state/debug/ZIP.
    """
    groq_status = check_groq_availability()
    if not groq_status["package_available"]:
        return None, groq_status["package_error"]
    if not groq_status["api_key_found"]:
        return None, groq_status["key_error"]
    try:
        from groq import Groq
        return Groq(api_key=get_groq_api_key()), None
    except Exception as e:
        return None, f"Groq client error: {e}"


# =============================================================================
# QWEN JSON CALLER — dengan error tracking jelas
# =============================================================================

def _call_groq_chat(
    client: Any,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Any:
    """
    Panggil Groq chat completions dengan fallback otomatis.

    Coba dulu dengan reasoning_effort="none" untuk menekan <think> tags.
    Jika SDK/model tidak mendukung parameter tersebut, ulangi tanpa parameter itu.
    """
    # Attempt 1: dengan reasoning_effort="none"
    try:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort="none",
        )
    except (TypeError, Exception) as e:
        err_str = str(e).lower()
        # Jika error karena parameter tidak dikenal, fallback tanpa reasoning_effort
        if "reasoning_effort" in err_str or isinstance(e, TypeError):
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raise  # Re-raise jika error lain (network, auth, dll)


def _clean_llm_response(raw: str) -> str:
    """
    Bersihkan response LLM dari thinking tags, markdown fences,
    dan teks non-JSON lainnya. Return string yang siap di-parse JSON.
    """
    # Hapus <think>...</think> (greedy, bisa multi-line)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw)
    # Hapus sisa tag <think> yang tidak tertutup (Qwen kadang terpotong)
    cleaned = re.sub(r"<think>[\s\S]*", "", cleaned)
    # Hapus markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned)
    # Hapus baris kosong berlebih
    cleaned = cleaned.strip()
    return cleaned


def _extract_json_object(text: str) -> Optional[Dict]:
    """
    Ekstrak objek JSON pertama yang valid dari teks campuran.
    Menggunakan pendekatan bracket-matching untuk menemukan { ... } terluar.
    """
    # Coba parse langsung dulu
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Cari objek JSON dengan bracket matching
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    # Mungkin ada objek JSON lain setelahnya, lanjut cari
                    start = text.find("{", i + 1)
                    if start == -1:
                        return None
                    depth = 0

    return None


def call_qwen_json(
    prompt: str,
    model_name: str,
    max_tokens: int = 500,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Panggil Groq API dan parse response sebagai JSON.

    Return: (data_dict, error_string)
    - Jika berhasil: (dict, None)
    - Jika gagal: (None, "pesan error yang jelas")

    raw_preview (max 500 char) disimpan untuk debug, tanpa API key.
    """
    client, client_err = get_groq_client()
    if client is None:
        return None, client_err

    # System message: paksa output JSON saja, tanpa penjelasan/thinking
    system_msg = {
        "role": "system",
        "content": (
            "You are a strict JSON generator. "
            "Return ONLY valid compact JSON. "
            "Do NOT include markdown, explanation, thinking text, "
            "or any text before/after the JSON object. "
            "NEVER use <think> tags."
        ),
    }
    user_msg = {"role": "user", "content": prompt}

    raw_response: Optional[str] = None
    try:
        resp = _call_groq_chat(
            client=client,
            model_name=model_name,
            messages=[system_msg, user_msg],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        raw_response = resp.choices[0].message.content or ""
    except Exception as e:
        return None, f"Groq API error: {type(e).__name__}: {str(e)[:200]}"

    if not raw_response.strip():
        return None, "Groq mengembalikan response kosong."

    # Bersihkan thinking tags, markdown fences, dll
    cleaned = _clean_llm_response(raw_response)

    if not cleaned:
        raw_preview = raw_response[:500]
        return None, f"Response hanya berisi thinking tags. Raw (500 char): {raw_preview!r}"

    # Ekstrak objek JSON (dengan bracket-matching yang robust)
    data = _extract_json_object(cleaned)
    if data is not None:
        return data, None

    # Gagal parse — simpan preview untuk debug
    raw_preview = raw_response[:500] if raw_response else ""
    return None, f"JSON parse error. Raw (500 char): {raw_preview!r}"


# =============================================================================
# QWEN VALIDATOR / REWRITER
# =============================================================================

def qwen_validate_or_rewrite_question(
    question: str,
    answer: str,
    context: str,
    question_type: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Validator/rewriter pertanyaan via Qwen.

    Qwen TIDAK mengubah jawaban benar. Hanya memperbaiki pertanyaan.
    Jawaban benar tetap dari pick_answer() (dari dokumen).

    Return:
    {
        "valid": bool,
        "question": str,          -- pertanyaan final
        "reason": str,
        "source": "qwen" | "heuristic_fallback",
        "qwen_used": bool,
        "qwen_error": str,        -- kosong jika berhasil
    }
    """
    prompt = (
        "Kamu adalah validator soal pilihan ganda Bahasa Indonesia.\n\n"
        f"Konteks dari dokumen:\n\"\"\"{context[:600]}\"\"\"\n\n"
        f"Pertanyaan: {question}\n"
        f"Jawaban benar: {answer}\n"
        f"Tipe pertanyaan: {question_type}\n\n"
        "Tugasmu:\n"
        "1. Cek apakah pertanyaan bisa dijawab dari konteks di atas.\n"
        "2. Cek apakah jawaban sesuai dengan isi konteks.\n"
        "3. Jika pertanyaan kurang baik, rewrite agar lebih jelas.\n"
        "4. JANGAN mengubah jawaban benar.\n"
        "5. JANGAN membuat fakta baru yang tidak ada di konteks.\n"
        "6. Jika pertanyaan sudah baik, pertahankan.\n\n"
        "Balas HANYA dengan JSON (tanpa penjelasan, tanpa markdown):\n"
        '{"valid": true, "question": "pertanyaan final di sini", "reason": "alasan singkat"}'
    )

    data, err = call_qwen_json(prompt, model_name, max_tokens=300)

    if data is not None and isinstance(data.get("question"), str) and data["question"].strip():
        return {
            "valid":      bool(data.get("valid", True)),
            "question":   data["question"].strip(),
            "reason":     data.get("reason", ""),
            "source":     "qwen",
            "qwen_used":  True,
            "qwen_error": "",
        }

    # Fallback ke heuristic jika Qwen gagal, dengan error yang jelas
    heuristic_result = heuristic_validate_question(question, answer, context, question_type)
    return {
        "valid":      heuristic_result.get("valid", True),
        "question":   heuristic_result.get("question", question),
        "reason":     heuristic_result.get("reason", ""),
        "source":     "heuristic_fallback",
        "qwen_used":  False,
        "qwen_error": err or "Qwen tidak berhasil dipanggil.",
    }


# =============================================================================
# QWEN DISTRACTOR GENERATOR
# =============================================================================

def _token_overlap(a: str, b: str) -> float:
    """Similarity sederhana berbasis token overlap untuk filter distractor mirip."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _is_safe_distractor(candidate: str, correct: str, existing: List[str]) -> bool:
    """
    Cek apakah kandidat distractor aman untuk dipakai:
    - Bukan jawaban benar (overlap < 0.7)
    - Tidak duplikat dengan yang sudah ada
    - Tidak mengandung noise (header/footer buku)
    - Tidak terlalu panjang (> 30 kata)
    - Tidak pakai frasa terlarang
    """
    c = clean_option_text(candidate)
    if not c or len(c) < 3:
        return False

    low = c.lower()
    words = c.split()

    if len(words) > 30:
        return False

    # Token overlap dengan jawaban benar
    if _token_overlap(c, correct) >= 0.7:
        return False

    # Duplikat dengan yang sudah ada
    for ex in existing:
        if _token_overlap(c, ex) >= 0.8:
            return False

    # Noise patterns
    noise = [
        "bahan ajar aplikasi komputer", "frizka fitriana",
        "daftar isi", "kata pengantar", "hak cipta",
        "semua jawaban", "tidak ada jawaban", "informasi tidak disebutkan",
        "jawaban belum dapat",
    ]
    if any(n in low for n in noise):
        return False

    return is_clean_distractor(c)


def qwen_generate_distractors(
    question: str,
    correct_answer: str,
    context: str,
    question_type: str,
    model_name: str,
) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Buat 3 distractor via Qwen.

    Return: (list_distractors, error_string)
    - Berhasil: (["d1","d2","d3"], None)
    - Gagal: (None, "pesan error")
    """
    prompt = (
        "Kamu adalah pembuat soal pilihan ganda Bahasa Indonesia yang ahli.\n\n"
        f"Konteks dari dokumen:\n\"\"\"{context[:600]}\"\"\"\n\n"
        f"Pertanyaan: {question}\n"
        f"Jawaban benar: {correct_answer}\n"
        f"Tipe pertanyaan: {question_type}\n\n"
        "Buat 3 pilihan jawaban yang SALAH tapi MENGECOH.\n"
        "Syarat setiap distractor:\n"
        "- Salah secara faktual berdasarkan konteks\n"
        "- Masih satu topik dengan jawaban benar\n"
        "- Tipe jawaban sama (panjang mirip, format mirip)\n"
        "- JANGAN gunakan 'bukan', 'tidak', 'semua di atas', 'informasi tidak disebutkan'\n"
        "- Bukan parafrase jawaban benar (overlap kata < 60%)\n"
        "- Tidak duplikat satu sama lain\n"
        "- Maksimal 25 kata per distractor\n\n"
        "Balas HANYA dengan JSON (tanpa penjelasan, tanpa markdown):\n"
        '{"distractors": ["distractor1", "distractor2", "distractor3"]}'
    )

    data, err = call_qwen_json(prompt, model_name, max_tokens=350)

    if data is not None and isinstance(data.get("distractors"), list):
        raw_list = data["distractors"]
        valid: List[str] = []
        for d in raw_list:
            if not isinstance(d, str):
                continue
            d = clean_option_text(d.strip())
            if _is_safe_distractor(d, correct_answer, valid):
                valid.append(d)
            if len(valid) >= 3:
                break
        if valid:
            return valid[:3], None
        return None, "Qwen berhasil dipanggil tapi semua distractor tidak lolos validasi."

    return None, err or "Qwen tidak mengembalikan data distractor yang valid."


def build_heuristic_distractors(
    correct: str,
    context: str,
    question_type: str,
    all_answers: List[str],
    n: int = 3,
) -> List[str]:
    """
    Distractor heuristic fallback yang aman.

    Prioritas:
    1. generate_heuristic_distractors dari preprocess.py (berbasis pola context)
    2. Ambil dari all_answers (jawaban benar soal lain)
    3. Pad minimal dengan frasa yang wajar, bukan "semua jawaban benar"

    Semua kandidat difilter dengan _is_safe_distractor.
    """
    # Tahap 1: dari preprocess
    candidates = []
    raw = generate_heuristic_distractors(correct, context, question_type, n=n + 5)
    for c in raw:
        c = clean_option_text(c)
        if _is_safe_distractor(c, correct, candidates):
            candidates.append(c)
        if len(candidates) >= n:
            return candidates

    # Tahap 2: dari pool jawaban soal lain
    shuffled = list(all_answers)
    random.shuffle(shuffled)
    for c in shuffled:
        c = clean_option_text(c)
        if _is_safe_distractor(c, correct, candidates):
            candidates.append(c)
        if len(candidates) >= n:
            return candidates

    # Tahap 3: last resort (hanya jika benar-benar tidak ada pilihan lain)
    last_resort = [
        "Pilihan ini tidak tersedia",
        "Belum dapat ditentukan dari konteks",
        "Tidak ada informasi yang mendukung",
    ]
    for lr in last_resort:
        if len(candidates) >= n:
            break
        if _is_safe_distractor(lr, correct, candidates):
            candidates.append(lr)

    return candidates[:n]


# =============================================================================
# PARSING HELPERS
# =============================================================================

def _is_noise_page(text: str) -> bool:
    """Deteksi halaman/slide/section non-konten."""
    if not text or not text.strip():
        return True
    low = text.lower().strip()
    words = low.split()
    if len(words) < 15:
        return True
    first_50 = " ".join(words[:50])
    noise_kw = [
        "kata pengantar","daftar isi","daftar pustaka","daftar gambar",
        "daftar tabel","ucapan terima kasih","prakata","sanwacana",
        "persembahan","daftar singkatan","lembar pengesahan","halaman judul","abstrak",
    ]
    if any(kw in first_50 for kw in noise_kw):
        return True
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        toc_count = sum(1 for l in lines if _TOC_LINE_RE.match(l))
        if toc_count / len(lines) > 0.35:
            return True
    gratitude_kw = [
        "mengucapkan terima kasih","terima kasih kepada","rasa syukur",
        "puji syukur","alhamdulillah","segala puji","penyusun mengucapkan","penulis mengucapkan",
    ]
    if sum(1 for kw in gratitude_kw if kw in low) >= 2:
        return True
    cover_kw = [
        "program studi","mata kuliah","disusun oleh","tim pengampu",
        "tahun ajaran","semester genap","semester ganjil","universitas","fakultas","nim :","nip :",
    ]
    if sum(1 for kw in cover_kw if kw in low) >= 3:
        return True
    if _PUBLICATION_META_RE.search(low):
        return True
    return False


def parse_pdf(b: bytes) -> Tuple[str, List[Dict]]:
    units: List[Dict] = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            page_texts = []
            for i, page in enumerate(pdf.pages):
                t = page.extract_text() or ""
                is_noise = _is_noise_page(t)
                units.append({"unit_id":i,"label":f"Halaman {i+1}","text":t,
                               "word_count":len(t.split()),"char_count":len(t),
                               "removed_reason":"noise_page" if is_noise else ""})
                if not is_noise and t.strip():
                    page_texts.append(t)
            if page_texts:
                cleaned = remove_repeated_headers_footers(page_texts, min_ratio=0.30)
                return "\n\n".join(p for p in cleaned if p.strip()), units
        return "[Error PDF: semua halaman difilter sebagai noise]", units
    except Exception as e:
        return f"[Error PDF: {e}]", units


def parse_pptx(b: bytes) -> Tuple[str, List[Dict]]:
    units: List[Dict] = []
    try:
        from pptx import Presentation
        prs = Presentation(io.BytesIO(b))
        slide_texts = []
        for i, slide in enumerate(prs.slides):
            shapes_with_text = []
            for shape in slide.shapes:
                if shape.has_text_frame and hasattr(shape, "text_frame"):
                    top  = shape.top  if getattr(shape,"top", None)  is not None else 0
                    left = shape.left if getattr(shape,"left", None) is not None else 0
                    tf = getattr(shape, "text_frame")
                    if tf and tf.text.strip():
                        shapes_with_text.append((top, left, tf.text.strip()))
            shapes_with_text.sort(key=lambda x: (x[0], x[1]))
            slide_text = "\n".join(s[2] for s in shapes_with_text)
            is_noise = _is_noise_page(slide_text) if slide_text.strip() else True
            units.append({"unit_id":i,"label":f"Slide {i+1}","text":slide_text,
                           "word_count":len(slide_text.split()),"char_count":len(slide_text),
                           "removed_reason":"noise_page" if is_noise else ""})
            if not is_noise and slide_text.strip():
                slide_texts.append(slide_text)
        if slide_texts:
            cleaned = remove_repeated_headers_footers(slide_texts, min_ratio=0.25)
            return "\n\n".join(p for p in cleaned if p.strip()), units
        return "[Error PPTX: semua slide difilter sebagai noise]", units
    except Exception as e:
        return f"[Error PPTX: {e}]", units


def parse_docx(b: bytes) -> Tuple[str, List[Dict]]:
    units: List[Dict] = []
    try:
        from docx import Document
        doc = Document(io.BytesIO(b))
        sections: List[str] = []
        current: List[str] = []
        sec_idx = 0
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style = para.style.name if para.style and para.style.name else ""
            if "Heading 1" in style or "Heading 2" in style:
                if current:
                    sec_text = "\n".join(current)
                    is_noise = _is_noise_page(sec_text)
                    units.append({"unit_id":sec_idx,"label":f"Section {sec_idx+1}","text":sec_text,
                                   "word_count":len(sec_text.split()),"char_count":len(sec_text),
                                   "removed_reason":"noise_page" if is_noise else ""})
                    if not is_noise:
                        sections.append(sec_text)
                    sec_idx += 1
                current = [text]
            else:
                current.append(text)
        if current:
            sec_text = "\n".join(current)
            is_noise = _is_noise_page(sec_text)
            units.append({"unit_id":sec_idx,"label":f"Section {sec_idx+1}","text":sec_text,
                           "word_count":len(sec_text.split()),"char_count":len(sec_text),
                           "removed_reason":"noise_page" if is_noise else ""})
            if not is_noise:
                sections.append(sec_text)
        if sections:
            return "\n\n".join(sections), units
        all_text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        if not _is_noise_page(all_text):
            return all_text, units
        return "[Error DOCX: konten tidak terdeteksi]", units
    except Exception as e:
        return f"[Error DOCX: {e}]", units


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_aqg_model():
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    try:
        tokenizer = T5Tokenizer.from_pretrained(HF_MODEL_REPO)
        model = T5ForConditionalGeneration.from_pretrained(HF_MODEL_REPO, torch_dtype=torch.float32)
        model.to(DEVICE).eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)


# =============================================================================
# GENERATE QUESTION
# =============================================================================

def clean_generated_question(raw_q: str) -> str:
    """Bersihkan pertanyaan hasil generate model."""
    q = (raw_q or "").strip()
    if not q:
        return ""
    words = q.split()
    dedup = []
    for w in words:
        if dedup and dedup[-1].lower().strip(".,?!") == w.lower().strip(".,?!"):
            continue
        dedup.append(w)
    q = " ".join(dedup)
    if q:
        q = q[0].upper() + q[1:]
    q = q.rstrip(" .,!;:")
    if not q.endswith("?"):
        q += "?"
    q = re.sub(r"\s+([?.!,;:])", r"\1", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def generate_question_with_model(answer: str, question_type: str, focused_context: str, tokenizer, model) -> str:
    """
    Generate pertanyaan dengan model IndoT5-QTYPE.
    Format input sesuai training: generate question: question_type: {qtype} answer: {answer} context: {ctx}
    """
    input_text = (
        f"generate question: question_type: {question_type} "
        f"answer: {answer.strip()} "
        f"context: {focused_context.strip()}"
    )
    try:
        inputs = tokenizer(
            input_text, max_length=MAX_INPUT_LEN,
            truncation=True, padding=False, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **GEN_CONFIG,
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return result if result else ""
    except Exception:
        return ""


# =============================================================================
# BUILD CHOICES — dengan tracking Qwen per soal
# =============================================================================

def build_choices(
    questions: List[Dict],
    qwen_enabled: bool,
    qwen_model: str,
) -> Tuple[List[List[str]], List[int], Dict[str, Any]]:
    """
    Buat 4 pilihan jawaban per soal: 1 benar (dari dokumen) + 3 distractor.

    Jawaban benar = q["answer"] dari pick_answer(), TIDAK boleh diubah oleh Qwen.
    Distractor: coba Qwen jika enabled, fallback ke heuristic.

    Return:
    - choices_all: list of list[str]
    - correct_all: list[int]          -- index jawaban benar (0-3)
    - qwen_tracking: dict berisi statistik Qwen yang jujur
    """
    # Cek Qwen sekali di awal, hemat API call
    qwen_ready = False
    qwen_init_error = ""
    if qwen_enabled:
        groq_status = check_groq_availability()
        if groq_status["package_available"] and groq_status["api_key_found"]:
            qwen_ready = True
        else:
            qwen_init_error = groq_status.get("package_error") or groq_status.get("key_error", "")

    # Pool jawaban benar semua soal — untuk fallback distractor
    all_answers = [
        clean_option_text(q.get("answer", ""))
        for q in questions
        if clean_option_text(q.get("answer", ""))
    ]

    choices_all: List[List[str]] = []
    correct_all: List[int] = []

    # Tracking Qwen per soal
    qwen_tracking: Dict[str, Any] = {
        "qwen_enabled_by_user":   qwen_enabled,
        "groq_package_available": check_groq_availability()["package_available"],
        "groq_api_key_found":     check_groq_availability()["api_key_found"],
        "qwen_model":             qwen_model,
        "qwen_ready":             qwen_ready,
        "qwen_init_error":        qwen_init_error,
        "distractor_qwen_used":   0,
        "distractor_heuristic_fallback": 0,
        "last_errors":            [],
    }

    for q in questions:
        correct  = clean_option_text(q.get("answer", ""))
        context  = q.get("focused_context") or q.get("context", "")
        question = q.get("question", "")
        qtype    = q.get("question_type", "apa")

        distractors: List[str] = []
        distractor_source = "heuristic_fallback"
        distractor_error  = ""

        # Coba Qwen
        if qwen_ready:
            d_list, d_err = qwen_generate_distractors(question, correct, context, qtype, qwen_model)
            if d_list and len(d_list) >= 1:
                distractors = d_list
                distractor_source = "qwen"
                qwen_tracking["distractor_qwen_used"] += 1
            else:
                distractor_error = d_err or "Qwen gagal menghasilkan distractor valid."
                qwen_tracking["last_errors"].append(f"Distractor soal '{question[:40]}': {distractor_error}")
                qwen_tracking["distractor_heuristic_fallback"] += 1
        else:
            qwen_tracking["distractor_heuristic_fallback"] += 1

        # Heuristic fallback / tambah jika Qwen kurang dari 3
        if len(distractors) < 3:
            hd = build_heuristic_distractors(correct, context, qtype, all_answers, n=3)
            for d in hd:
                if _is_safe_distractor(d, correct, distractors):
                    distractors.append(d)
                if len(distractors) >= 3:
                    break

        # Pastikan tepat 3
        if len(distractors) < 3:
            fallback_phrases = [
                "Pilihan ini tidak tersedia",
                "Belum dapat ditentukan dari konteks",
                "Tidak ada informasi yang mendukung",
            ]
            fb_idx = 0
            while len(distractors) < 3:
                distractors.append(fallback_phrases[fb_idx % len(fallback_phrases)])
                fb_idx += 1

        distractors = distractors[:3]

        # Simpan source tracking ke soal
        q["distractor_source"] = distractor_source
        q["distractor_error"]  = distractor_error

        # Susun 4 opsi, acak, simpan index
        opts = distractors + [correct]
        random.shuffle(opts)
        try:
            correct_idx = opts.index(correct)
        except ValueError:
            opts[-1] = correct
            correct_idx = len(opts) - 1

        choices_all.append(opts)
        correct_all.append(correct_idx)

    # Potong last_errors agar tidak terlalu besar di ZIP
    qwen_tracking["last_errors"] = qwen_tracking["last_errors"][:20]

    return choices_all, correct_all, qwen_tracking


# =============================================================================
# DEBUG ZIP BUILDER
# =============================================================================

def build_debug_zip(session_data: Dict) -> bytes:
    """
    Buat ZIP debug lengkap.

    File yang dihasilkan:
    00_parsing_summary.json
    01_raw_units.csv
    02_removed_units.csv
    10_cleaned.txt  ...  16_restored.txt
    20_pipeline_stages.json
    21_chunks.csv
    22_question_attempts.csv
    30_questions.csv
    31_questions_full.json
    32_choices.json
    33_qwen_status.json
    34_qwen_errors.csv
    """
    import csv

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

        def add_text(name: str, content: str):
            if content:
                zf.writestr(name, content.encode("utf-8"))

        def add_json(name: str, obj: Any):
            try:
                # Konversi dict keys ke string dan pastikan JSON serializable
                safe_obj = json.loads(json.dumps(obj, default=str))
                zf.writestr(name, json.dumps(safe_obj, ensure_ascii=False, indent=2).encode("utf-8"))
            except Exception:
                pass

        def add_csv_from_list(name: str, rows: List[Dict], fieldnames: List[str]):
            if not rows:
                return
            out = io.StringIO()
            w = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
            zf.writestr(name, out.getvalue().encode("utf-8"))

        # 00 Parsing summary
        units = session_data.get("units", [])
        add_json("00_parsing_summary.json", {
            "file_name":       session_data.get("file_name", ""),
            "file_type":       session_data.get("file_type", ""),
            "total_units":     len(units),
            "removed_units":   sum(1 for u in units if u.get("removed_reason")),
            "valid_units":     sum(1 for u in units if not u.get("removed_reason")),
            "raw_text_words":  len(session_data.get("raw_text", "").split()),
            "chunks_skipped_low_score": session_data.get("stats_skipped_low_score", 0),
            "question_type_dist": dict(session_data.get("stats_question_types", {})),
            "reject_reasons": dict(session_data.get("stats_reject_reasons", {})),
        })

        # 01-02 Units
        add_csv_from_list("01_raw_units.csv", units,
            ["unit_id","label","word_count","char_count","removed_reason","text"])
        removed = [u for u in units if u.get("removed_reason")]
        add_csv_from_list("02_removed_units.csv", removed,
            ["unit_id","label","word_count","removed_reason"])

        # 10-16 Pipeline text stages
        pipeline = session_data.get("pipeline", {})
        add_text("10_cleaned.txt",           pipeline.get("cleaned", ""))
        add_text("11_noise_filtered.txt",     pipeline.get("noise_filtered", ""))
        add_text("12_task_block_removed.txt", pipeline.get("task_block_removed", ""))
        add_text("13_reconstructed.txt",      pipeline.get("reconstructed", ""))
        add_text("14_normalized.txt",         pipeline.get("normalized", ""))
        add_text("15_expanded.txt",           pipeline.get("expanded", ""))
        add_text("16_restored.txt",           pipeline.get("restored", ""))
        add_json("20_pipeline_stages.json",   pipeline.get("pipeline_stages", {}))

        # 21 Chunks
        chunks = pipeline.get("chunks", [])
        add_csv_from_list("21_chunks.csv", chunks,
            ["chunk_id","section","word_count","n_tokens","n_sents","content_score","method","text"])

        # 22 Question attempts
        attempts = session_data.get("question_attempts", [])
        add_csv_from_list("22_question_attempts.csv", attempts, [
            "chunk_id","answer","question_type","focused_context","score",
            "raw_question","cleaned_question",
            "qwen_validator_used","qwen_validator_error","validator_source","validator_reason",
            "accepted",
        ])

        # 30-32 Questions & choices
        questions = session_data.get("questions", [])
        add_csv_from_list("30_questions.csv", questions,
            ["chunk_id","question_type","question","answer","focused_context","context",
             "validator_source","distractor_source","distractor_error"])
        add_json("31_questions_full.json", questions)

        choices_data = []
        for i, q in enumerate(questions):
            choices_all = session_data.get("quiz_choices", [])
            correct_all = session_data.get("quiz_correct", [])
            if i < len(choices_all):
                choices_data.append({
                    "soal_idx":      i,
                    "question":      q.get("question", ""),
                    "correct_answer":q.get("answer", ""),
                    "choices":       choices_all[i],
                    "correct_idx":   correct_all[i] if i < len(correct_all) else -1,
                    "distractor_source": q.get("distractor_source", ""),
                })
        add_json("32_choices.json", choices_data)

        # 33 Qwen status — TIDAK menyimpan API key
        qt = session_data.get("qwen_tracking", {})
        add_json("33_qwen_status.json", {
            "qwen_enabled_by_user":          qt.get("qwen_enabled_by_user", False),
            "groq_api_key_found":            qt.get("groq_api_key_found", False),
            "groq_package_available":        qt.get("groq_package_available", False),
            "qwen_model":                    qt.get("qwen_model", ""),
            "qwen_ready":                    qt.get("qwen_ready", False),
            "qwen_init_error":               qt.get("qwen_init_error", ""),
            "distractor_qwen_used":          qt.get("distractor_qwen_used", 0),
            "distractor_heuristic_fallback": qt.get("distractor_heuristic_fallback", 0),
            "validator_qwen_used":           session_data.get("validator_qwen_used", 0),
            "validator_heuristic_fallback":  session_data.get("validator_heuristic_fallback", 0),
            "last_errors":                   qt.get("last_errors", []),
        })

        # 34 Qwen errors CSV
        errors = qt.get("last_errors", [])
        if errors:
            out = io.StringIO()
            w = csv.writer(out)
            w.writerow(["error_message"])
            for err in errors:
                w.writerow([err])
            zf.writestr("34_qwen_errors.csv", out.getvalue().encode("utf-8"))

    buf.seek(0)
    return buf.read()


# =============================================================================
# SESSION STATE DEFAULTS
# =============================================================================

defaults = {
    "phase":                    "upload",
    "questions":                [],
    "question_attempts":        [],
    "quiz_choices":             [],
    "quiz_correct":             [],
    "quiz_idx":                 0,
    "quiz_answers":             {},
    "quiz_score":               0,
    "raw_text":                 "",
    "file_name":                "",
    "file_type":                "",
    "max_q":                    8,
    "units":                    [],
    "pipeline":                 {},
    "qwen_tracking":            {},
    "validator_qwen_used":      0,
    "validator_heuristic_fallback": 0,
    "debug_zip":                None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# PHASE: UPLOAD
# =============================================================================

if st.session_state.phase == "upload":

    st.markdown('<div class="sec-label">Upload materi kuliah</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop atau klik untuk upload",
        type=["pdf","docx","pptx"],
        label_visibility="collapsed",
    )

    if uploaded:
        ftype  = uploaded.name.rsplit(".",1)[-1].lower()
        fbytes = uploaded.read()

        col_q, col_btn = st.columns([3,1])
        with col_q:
            max_q = st.slider("Jumlah soal", min_value=3, max_value=20, value=8,
                              help="Jumlah aktual bisa lebih sedikit jika chunk tidak cukup.")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("Generate →", use_container_width=True, type="primary")

        if run_btn:
            with st.spinner("📄 Membaca dan memfilter file..."):
                parsers = {"pdf":parse_pdf, "docx":parse_docx, "pptx":parse_pptx}
                raw_text, units = parsers[ftype](fbytes)

            if raw_text.startswith("[Error"):
                st.error(raw_text)
            else:
                st.session_state.raw_text  = raw_text
                st.session_state.file_name = uploaded.name
                st.session_state.file_type = ftype
                st.session_state.max_q     = max_q
                st.session_state.units     = units
                st.session_state.phase     = "generating"
                st.rerun()


# =============================================================================
# PHASE: GENERATING
# =============================================================================

elif st.session_state.phase == "generating":

    st.markdown("""
    <div class="card">
        <div style="font-size:1.05rem;font-weight:600;color:var(--text-dark);margin-bottom:0.3rem;">⚙️ Sedang memproses materi...</div>
        <div style="font-size:0.82rem;color:var(--text-soft);">Tahapan: parsing → preprocessing → chunking → generate soal.</div>
    </div>""", unsafe_allow_html=True)

    # Cek Qwen sebelum generate — agar status jelas di awal
    groq_availability = check_groq_availability()
    qwen_will_run = use_qwen and groq_availability["package_available"] and groq_availability["api_key_found"]

    if use_qwen and not qwen_will_run:
        err_msg = groq_availability.get("package_error") or groq_availability.get("key_error","")
        st.warning(f"⚠️ Qwen diaktifkan tapi tidak bisa dipakai: {err_msg}")

    # Load model
    with st.spinner("Memuat model IndoT5-QTYPE dari HuggingFace..."):
        tokenizer, model, model_err = load_aqg_model()

    if model_err:
        st.error(f"❌ Gagal memuat model: {model_err}")
        if st.button("← Kembali ke Upload"):
            st.session_state.phase = "upload"
            st.rerun()
        st.stop()

    # Preprocessing
    with st.spinner("Menjalankan pipeline preprocessing (9 tahap)..."):
        pipeline = full_preprocess_pipeline(st.session_state.raw_text)
    st.session_state.pipeline = pipeline

    chunks = [ch for ch in pipeline.get("chunks",[]) if ch.get("text","").strip()]
    chunks.sort(key=lambda x: x.get("content_score",0), reverse=True)

    max_q = st.session_state.max_q
    pb = st.progress(0.0, text="Generating soal...")

    results:  List[Dict] = []
    attempts: List[Dict] = []
    seen_q:   set = set()
    seen_ans: set = set()

    # Counter Qwen validator per soal
    validator_qwen_used        = 0
    validator_heuristic_used   = 0
    qwen_validator_errors: List[str] = []
    
    # Stats untuk debug
    from collections import Counter
    stats_skipped_low_score = 0
    stats_question_types = Counter()
    stats_reject_reasons = Counter()

    def _norm_q(q: str) -> str:
        q = q.lower().strip()
        q = re.sub(r"\b(apakah|apakan)\b","apa",q)
        q = re.sub(r"\b(bagaimanakah)\b","bagaimana",q)
        q = re.sub(r"\b(mengapakah)\b","mengapa",q)
        q = re.sub(r"\b(siapakah)\b","siapa",q)
        q = re.sub(r"\b(dimanakah|di\s+manakah)\b","dimana",q)
        q = re.sub(r"\b(kapankah)\b","kapan",q)
        q = re.sub(r"\b(berapakah)\b","berapa",q)
        return re.sub(r"\s+"," ",q).strip()

    for ch in chunks:
        if len(results) >= max_q:
            break

        text = ch.get("text","").strip()
        if not text:
            continue
            
        if ch.get("content_score", 0) < 1:
            stats_skipped_low_score += 1
            continue

        ans_data = pick_answer(text)
        if ans_data is None:
            stats_reject_reasons["No valid answer/pattern in pick_answer"] += 1
            continue

        answer        = ans_data["answer"]
        question_type = ans_data["question_type"]
        focused_ctx   = ans_data.get("focused_context") or text
        target_term   = ans_data.get("target_term","")
        score         = ans_data.get("score", 0.0)
        reason        = ans_data.get("reason", "")

        if not is_clean_distractor(answer):
            stats_reject_reasons["Answer not clean distractor"] += 1
            continue
        if score < 0:
            stats_reject_reasons[f"Answer score < 0 ({reason})"] += 1
            continue

        ans_key = _norm_q(answer[:60])
        if ans_key in seen_ans:
            continue

        # Generate question
        raw_q     = generate_question_with_model(answer, question_type, focused_ctx, tokenizer, model)
        cleaned_q = clean_generated_question(raw_q)

        attempt: Dict[str,Any] = {
            "chunk_id":              ch.get("chunk_id",-1),
            "answer":                answer,
            "question_type":         question_type,
            "focused_context":       focused_ctx,
            "score":                 score,
            "raw_question":          raw_q,
            "cleaned_question":      cleaned_q,
            "qwen_validator_used":   False,
            "qwen_validator_error":  "",
            "validator_source":      "none",
            "validator_reason":      "",
            "accepted":              False,
        }

        if not cleaned_q:
            attempt["validator_reason"] = "Pertanyaan kosong setelah dibersihkan"
            attempts.append(attempt)
            stats_reject_reasons["Generated question empty"] += 1
            continue
            
        short_q_re = re.compile(r"^(apa itu|apa fungsi|apa manfaat|apa tujuan|siapa|kapan|berapa|di mana|bagaimana)\b", re.IGNORECASE)
        
        if len(cleaned_q) < 8 and not short_q_re.match(cleaned_q):
            attempt["validator_reason"] = "Pertanyaan terlalu pendek atau format kurang dikenali"
            attempts.append(attempt)
            stats_reject_reasons["Generated question too short/invalid"] += 1
            continue

        q_norm = _norm_q(cleaned_q)
        if q_norm in seen_q or q_norm.startswith("["):
            attempts.append(attempt)
            continue

        # Validator Qwen atau heuristic
        final_q = cleaned_q
        if qwen_will_run:
            val = qwen_validate_or_rewrite_question(
                cleaned_q, answer, focused_ctx, question_type, qwen_model_name
            )
            final_q                      = val["question"]
            attempt["qwen_validator_used"]  = val["qwen_used"]
            attempt["qwen_validator_error"] = val["qwen_error"]
            attempt["validator_source"]     = val["source"]
            attempt["validator_reason"]     = val["reason"]
            if val["qwen_used"]:
                validator_qwen_used += 1
            else:
                validator_heuristic_used += 1
                if val["qwen_error"]:
                    qwen_validator_errors.append(f"Validator '{cleaned_q[:40]}': {val['qwen_error']}")
        else:
            val = heuristic_validate_question(cleaned_q, answer, focused_ctx, question_type)
            if not val.get("valid", True):
                attempt.update({"validator_source":"heuristic","validator_reason":val.get("reason","")})
                attempts.append(attempt)
                continue
            final_q                      = val.get("question", cleaned_q)
            attempt["validator_source"]  = "heuristic"
            attempt["validator_reason"]  = val.get("reason","")
            validator_heuristic_used += 1

        attempt["accepted"] = True
        attempts.append(attempt)

        seen_q.add(_norm_q(final_q))
        seen_ans.add(ans_key)

        results.append({
            "chunk_id":       ch.get("chunk_id",-1),
            "question":       final_q,
            "question_type":  question_type,
            "answer":         answer,
            "focused_context":focused_ctx,
            "context":        text,
            "target_term":    target_term,
            "score":          score,
            "validator_source": attempt["validator_source"],
            "distractor_source": "",  # diisi saat build_choices
            "distractor_error":  "",
        })

        pb.progress(min(len(results)/max_q,1.0), text=f"Generating soal... {len(results)}/{max_q}")
        
        stats_question_types[question_type] += 1

    pb.empty()

    # Simpan statistik ke session state untuk ZIP
    st.session_state.stats_skipped_low_score = stats_skipped_low_score
    st.session_state.stats_question_types = stats_question_types
    st.session_state.stats_reject_reasons = stats_reject_reasons

    if not results:
        st.error("❌ Tidak ada soal yang berhasil di-generate. Coba upload materi yang lebih panjang.")
        if st.button("← Kembali"):
            st.session_state.phase = "upload"
            st.rerun()
        st.stop()

    # Build choices dengan tracking Qwen
    with st.spinner("Membuat pilihan jawaban..."):
        choices_all, correct_all, qwen_tracking = build_choices(
            results, use_qwen, qwen_model_name
        )

    # Tambahkan error validator ke tracking
    qwen_tracking["last_errors"] = (qwen_validator_errors + qwen_tracking.get("last_errors",[]))[:20]
    qwen_tracking["validator_qwen_used"] = validator_qwen_used
    qwen_tracking["validator_heuristic_fallback"] = validator_heuristic_used

    # Simpan ke session state
    st.session_state.questions                  = results
    st.session_state.question_attempts          = attempts
    st.session_state.quiz_choices               = choices_all
    st.session_state.quiz_correct               = correct_all
    st.session_state.quiz_idx                   = 0
    st.session_state.quiz_answers               = {}
    st.session_state.quiz_score                 = 0
    st.session_state.qwen_tracking              = qwen_tracking
    st.session_state.validator_qwen_used        = validator_qwen_used
    st.session_state.validator_heuristic_fallback = validator_heuristic_used

    # Build debug ZIP
    debug_data = {
        "file_name":                st.session_state.file_name,
        "file_type":                st.session_state.file_type,
        "raw_text":                 st.session_state.raw_text,
        "units":                    st.session_state.units,
        "pipeline":                 pipeline,
        "questions":                results,
        "question_attempts":        attempts,
        "quiz_choices":             choices_all,
        "quiz_correct":             correct_all,
        "qwen_tracking":            qwen_tracking,
        "validator_qwen_used":      validator_qwen_used,
        "validator_heuristic_fallback": validator_heuristic_used,
    }
    st.session_state.debug_zip = build_debug_zip(debug_data)
    st.session_state.phase = "popup"
    st.rerun()


# =============================================================================
# PHASE: POPUP
# =============================================================================

elif st.session_state.phase == "popup":
    n             = len(st.session_state.questions)
    max_requested = st.session_state.max_q
    qt            = st.session_state.qwen_tracking

    warn_html = ""
    if n < max_requested:
        warn_html = f"""<div style="font-size:0.78rem;color:#B45309;background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:0.5rem 0.9rem;margin-bottom:1.5rem;line-height:1.6;">
            ⚠️ Diminta <b>{max_requested} soal</b>, berhasil generate <b>{n} soal</b>. Upload materi lebih panjang untuk soal lebih banyak.</div>"""

    # ── Debug panel ──────────────────────────────────────────────
    if show_debug:
        st.markdown('<div class="sec-label">📊 Ringkasan Proses</div>', unsafe_allow_html=True)

        pipeline = st.session_state.pipeline
        stages   = pipeline.get("pipeline_stages", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Kata (raw)", stages.get("01_raw_words",0))
            st.metric("Paragraf awal", stages.get("03_raw_paragraphs",0))
        with col2:
            st.metric("Setelah cleaning", stages.get("02_cleaned_words",0))
            st.metric("Setelah filter", stages.get("05_filtered_paragraphs",0))
        with col3:
            st.metric("Chunks total", stages.get("08_chunks",0))
            st.metric("Soal berhasil", n)

        # ── Qwen Status Panel — jujur dan detail ─────────────────
        st.markdown('<div class="sec-label">🤖 Status Qwen</div>', unsafe_allow_html=True)

        pkg_ok  = qt.get("groq_package_available", False)
        key_ok  = qt.get("groq_api_key_found", False)
        enabled = qt.get("qwen_enabled_by_user", False)
        ready   = qt.get("qwen_ready", False)

        pkg_label  = "✅ tersedia"  if pkg_ok  else "❌ tidak tersedia"
        key_label  = "✅ terbaca"   if key_ok  else "❌ tidak terbaca"
        ready_lbl  = "✅ aktif"     if ready   else "⚠️ tidak aktif"
        en_label   = "Ya"           if enabled else "Tidak"

        d_qwen     = qt.get("distractor_qwen_used", 0)
        d_heur     = qt.get("distractor_heuristic_fallback", 0)
        v_qwen     = st.session_state.get("validator_qwen_used", 0)
        v_heur     = st.session_state.get("validator_heuristic_fallback", 0)

        st.markdown(f"""
        <div class="qwen-panel">
            <div class="qwen-row"><span class="qwen-key">Aktifkan Qwen (user)</span><span class="qwen-val">{en_label}</span></div>
            <div class="qwen-row"><span class="qwen-key">Groq package</span><span class="qwen-val {'qwen-ok' if pkg_ok else 'qwen-err'}">{pkg_label}</span></div>
            <div class="qwen-row"><span class="qwen-key">Groq API key</span><span class="qwen-val {'qwen-ok' if key_ok else 'qwen-err'}">{key_label}</span></div>
            <div class="qwen-row"><span class="qwen-key">Model Qwen</span><span class="qwen-val">{qt.get('qwen_model','—')}</span></div>
            <div class="qwen-row"><span class="qwen-key">Status keseluruhan</span><span class="qwen-val {'qwen-ok' if ready else 'qwen-warn'}">{ready_lbl}</span></div>
            <div style="border-top:1px solid var(--border-s);margin:0.5rem 0;"></div>
            <div class="qwen-row"><span class="qwen-key">Validator Qwen berhasil</span><span class="qwen-val">{v_qwen} / {v_qwen+v_heur} soal</span></div>
            <div class="qwen-row"><span class="qwen-key">Validator heuristic fallback</span><span class="qwen-val">{v_heur} soal</span></div>
            <div class="qwen-row"><span class="qwen-key">Distractor Qwen berhasil</span><span class="qwen-val">{d_qwen} / {d_qwen+d_heur} soal</span></div>
            <div class="qwen-row"><span class="qwen-key">Distractor heuristic fallback</span><span class="qwen-val">{d_heur} soal</span></div>
        </div>""", unsafe_allow_html=True)

        # Tampilkan error Qwen jika ada
        errs = qt.get("last_errors", [])
        if errs:
            with st.expander(f"⚠️ Error Qwen ({len(errs)} entri)", expanded=False):
                for i, err in enumerate(errs[:10], 1):
                    st.markdown(f"""<div style="font-size:0.78rem;color:var(--red-txt);margin-bottom:0.3rem;">{i}. {err}</div>""",
                    unsafe_allow_html=True)
        elif enabled and not ready:
            init_err = qt.get("qwen_init_error","")
            if init_err:
                st.error(f"Qwen tidak bisa dipakai: {init_err}")

        # Tabel unit dibuang
        removed_units = [u for u in st.session_state.units if u.get("removed_reason")]
        if removed_units:
            with st.expander(f"📋 Unit dibuang ({len(removed_units)})", expanded=False):
                import pandas as pd
                df_rem = pd.DataFrame(removed_units)[["unit_id","label","word_count","removed_reason"]]
                st.dataframe(df_rem, use_container_width=True, hide_index=True)

        # Tabel chunks
        chunks = pipeline.get("chunks", [])
        if chunks:
            with st.expander(f"🔢 Chunks ({len(chunks)})", expanded=False):
                import pandas as pd
                df_ch = pd.DataFrame(chunks)[["chunk_id","section","word_count","content_score","method"]]
                st.dataframe(df_ch, use_container_width=True, hide_index=True)

        # Tabel question attempts
        attempts = st.session_state.question_attempts
        if attempts:
            with st.expander(f"🔍 Question attempts ({len(attempts)})", expanded=False):
                import pandas as pd
                df_att = pd.DataFrame(attempts)[[
                    "chunk_id","cleaned_question","answer","question_type","score",
                    "qwen_validator_used","qwen_validator_error","validator_source","accepted"
                ]]
                st.dataframe(df_att, use_container_width=True, hide_index=True)

        # Tabel final questions dengan distractor source
        qs = st.session_state.questions
        if qs:
            with st.expander(f"📝 Soal final ({len(qs)})", expanded=False):
                import pandas as pd
                df_q = pd.DataFrame(qs)
                cols = ["question","answer","question_type","validator_source","distractor_source","distractor_error"]
                cols = [c for c in cols if c in df_q.columns]
                st.dataframe(df_q[cols], use_container_width=True, hide_index=True)

        if st.session_state.debug_zip:
            st.download_button(
                label="⬇️ Download Debug ZIP",
                data=st.session_state.debug_zip,
                file_name=f"aqg_debug_{st.session_state.file_name}.zip",
                mime="application/zip",
                use_container_width=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

    # ── Popup card ───────────────────────────────────────────────
    _, col_mid, _ = st.columns([1,3,1])
    with col_mid:
        st.markdown(f"""
        <div style="background:#FFF;border-radius:20px;padding:2.4rem 2.2rem 1.6rem;text-align:center;
                    box-shadow:0 24px 60px rgba(30,58,95,0.22);border-top:4px solid #4A90D9;
                    margin-top:1rem;animation:popIn 0.35s cubic-bezier(0.34,1.56,0.64,1);">
            <div style="font-size:2.8rem;margin-bottom:0.8rem;">🎉</div>
            <div style="font-family:'Lora',serif;font-size:1.5rem;font-weight:600;color:#1A2B45;margin-bottom:0.4rem;">Soal siap!</div>
            <p style="font-size:0.82rem;color:#6B87A8;line-height:1.6;margin-bottom:1.2rem;">
                File <b style="color:#1A2B45;">{st.session_state.file_name}</b><br>berhasil diproses menjadi soal pilihan ganda.
            </p>
            <div style="display:inline-flex;align-items:center;gap:0.4rem;background:#EFF6FF;color:#2563A8;
                        border:1.5px solid #DBEAFE;border-radius:99px;font-size:0.72rem;font-weight:700;
                        padding:0.28rem 1.1rem;margin-bottom:1.5rem;letter-spacing:0.04em;">
                ✦ &nbsp;{n} soal pilihan ganda berhasil di-generate
            </div>
            {warn_html}
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        if st.button("🚀 Mulai Kuis", use_container_width=True, type="primary"):
            st.session_state.phase = "quiz"
            st.rerun()
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        if st.button("↩ Upload ulang", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()


# =============================================================================
# PHASE: QUIZ
# =============================================================================

elif st.session_state.phase == "quiz":
    questions   = st.session_state.questions
    choices_all = st.session_state.quiz_choices
    correct_all = st.session_state.quiz_correct
    idx         = st.session_state.quiz_idx
    total       = len(questions)

    if idx >= total:
        st.session_state.phase = "result"
        st.rerun()

    q           = questions[idx]
    opts        = choices_all[idx]
    correct_idx = correct_all[idx]
    correct_str = opts[correct_idx]
    pct         = idx / total * 100

    st.markdown(f"""
    <div class="quiz-meta">
        <span class="quiz-meta-label">Progress kuis</span>
        <span class="quiz-meta-counter">{idx+1} / {total}</span>
    </div>
    <div class="prog-wrap"><div class="prog-fill" style="width:{pct:.1f}%"></div></div>
    """, unsafe_allow_html=True)

    qtype_label = q.get("question_type","apa").replace("_"," ").title()
    st.markdown(f"""
    <div class="card">
        <div class="quiz-q-label">Pertanyaan {idx+1:02d}</div>
        <div style="margin-bottom:0.6rem;"><span class="qtype-badge">{qtype_label}</span></div>
        <div class="quiz-q-text">{q['question']}</div>
    </div>""", unsafe_allow_html=True)

    # Radio dengan index=None — TIDAK ada placeholder kosong
    prev = st.session_state.quiz_answers.get(idx, {})
    prev_selected = prev.get("selected") if isinstance(prev, dict) else None

    if prev_selected and prev_selected in opts:
        default_idx = opts.index(prev_selected)
    else:
        default_idx = None  # Tidak ada yang terpilih secara default

    selected = st.radio(
        "Pilih jawaban:",
        options=opts,
        index=default_idx,
        key=f"quiz_radio_{idx}",
        label_visibility="collapsed",
    )

    # Konteks sumber
    with st.expander("🔍 Lihat konteks sumber"):
        ctx_display = q.get("focused_context") or q.get("context","")
        st.markdown(f"""<div style="font-size:0.82rem;color:var(--text-mid);line-height:1.6;">{ctx_display}</div>""",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_back, col_spacer, col_next = st.columns([1,2,1])
    with col_back:
        if idx > 0 and st.button("← Kembali", use_container_width=True):
            st.session_state.quiz_idx = idx - 1
            st.rerun()

    with col_next:
        btn_label = "Selesai ✓" if idx == total-1 else "Lanjut →"
        if st.button(btn_label, use_container_width=True, type="primary"):
            if selected is None:
                st.warning("⚠️ Pilih salah satu jawaban dulu sebelum lanjut.")
            else:
                is_correct = (opts.index(selected) == correct_idx)
                st.session_state.quiz_answers[idx] = {
                    "selected":   selected,
                    "correct":    correct_str,
                    "is_correct": is_correct,
                }
                if is_correct:
                    st.session_state.quiz_score += 1
                st.session_state.quiz_idx = idx + 1
                if st.session_state.quiz_idx >= total:
                    st.session_state.phase = "result"
                st.rerun()


# =============================================================================
# PHASE: RESULT
# =============================================================================

elif st.session_state.phase == "result":
    questions = st.session_state.questions
    answers   = st.session_state.quiz_answers
    score     = st.session_state.quiz_score
    total     = len(questions)
    pct       = round(score / total * 100) if total else 0
    grade     = "A" if pct >= 85 else "B" if pct >= 70 else "C" if pct >= 55 else "D"

    st.markdown(f"""
    <div class="score-wrap">
        <div class="score-big">{pct}%</div>
        <div class="score-label">Skor akhir · {score} dari {total} benar</div>
        <div class="score-grade">Grade {grade}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Rekap jawaban</div>', unsafe_allow_html=True)

    for i, q in enumerate(questions):
        ans_data = answers.get(i, {})
        usr      = ans_data.get("selected","(tidak dijawab)") if isinstance(ans_data,dict) else "(tidak dijawab)"
        correct  = ans_data.get("correct","-")                if isinstance(ans_data,dict) else "-"
        is_ok    = ans_data.get("is_correct",False)            if isinstance(ans_data,dict) else False
        cls      = "correct" if is_ok else "wrong"
        icon     = "✓" if is_ok else "✕"
        qtype    = q.get("question_type","apa").replace("_"," ").title()

        st.markdown(f"""
        <div class="result-card {cls}">
            <div class="rc-label" style="font-size:0.6rem;letter-spacing:0.14em;text-transform:uppercase;font-weight:700;margin-bottom:0.3rem;">
                {icon} Soal {i+1:02d} · {qtype}
            </div>
            <div style="font-family:'Lora',serif;font-size:0.92rem;color:var(--text-dark);margin-bottom:0.3rem;line-height:1.5;">{q['question']}</div>
            <div style="font-size:0.79rem;color:var(--text-mid);">Jawabanmu: <b>{usr}</b></div>
            <div style="font-size:0.79rem;color:var(--text-mid);margin-top:0.15rem;">Jawaban benar: <b>{correct}</b></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Ulangi Kuis", use_container_width=True):
            st.session_state.quiz_idx     = 0
            st.session_state.quiz_answers = {}
            st.session_state.quiz_score   = 0
            new_choices, new_correct, new_qt = build_choices(questions, use_qwen, qwen_model_name)
            st.session_state.quiz_choices    = new_choices
            st.session_state.quiz_correct    = new_correct
            st.session_state.qwen_tracking   = new_qt
            st.session_state.phase           = "quiz"
            st.rerun()
    with col2:
        if st.button("📤 Upload Materi Baru", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

    if show_debug and st.session_state.debug_zip:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Debug ZIP",
            data=st.session_state.debug_zip,
            file_name=f"aqg_debug_{st.session_state.file_name}.zip",
            mime="application/zip",
            use_container_width=True,
        )
