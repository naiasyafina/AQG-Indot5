"""
AQG — Automatic Question Generator
Single-page UX: Upload → Popup → Quiz (one by one)
Kelompok: Naia Syafina H. | Onalla Aldeanuva
"""

import os, re, io, random, warnings, json
import streamlit as st
import torch
from groq import Groq

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MAX_INPUT_LEN  = 1024
MAX_TARGET_LEN = 64
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
GEN_CONFIG     = dict(
    decoder_start_token_id = 0,
    eos_token_id           = 1,
    pad_token_id           = 0,
    use_cache              = True,
    max_length             = MAX_TARGET_LEN,
    min_length             = 5,
    num_beams              = 4,
    repetition_penalty     = 1.2,
    length_penalty         = 1.0,
    early_stopping         = True,
    no_repeat_ngram_size   = 3,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQG — Question Generator",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1E3A5F 0%, #2563A8 100%) !important;
    }
    [data-testid="stSidebar"] { color: #E8F4FF !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(147,196,238,0.25) !important; }
    </style>

    <div style="padding: 1.4rem 0.4rem 0.6rem;">
        <div style="font-size:0.6rem;letter-spacing:0.22em;text-transform:uppercase;
                    color:#93C4EE;font-weight:600;margin-bottom:0.5rem;">✦ NLP Project · 2026</div>
        <div style="font-family:'Georgia',serif;font-size:1.5rem;font-weight:700;
                    color:#FFFFFF;line-height:1.2;margin-bottom:0.3rem;">
            Automatic<br>Question<br>Generator
        </div>
        <div style="font-size:0.75rem;color:rgba(147,196,238,0.8);margin-top:0.4rem;">
            Fine-tuned IndoT5 untuk pembangkitan soal otomatis dari teks kuliah Bahasa Indonesia.
        </div>
    </div>
    <hr style="margin: 1.2rem 0;">

    <div style="font-size:0.6rem;letter-spacing:0.18em;text-transform:uppercase;
                color:#93C4EE;font-weight:600;margin-bottom:0.8rem;">👥 Anggota Tim</div>
    """, unsafe_allow_html=True)

    members = [
        ("Naia Syafina H.",       "A11.2024.15554"),
        ("Onalla Aldeanuva",      "A11.2024.15952"),
    ]
    for name, nim in members:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;margin-bottom:0.75rem;
                    background:rgba(255,255,255,0.07);border-radius:10px;
                    padding:0.6rem 0.8rem;">
            <span style="font-size:0.83rem;font-weight:600;color:#FFFFFF;">{name}</span>
            <span style="font-size:0.7rem;color:rgba(147,196,238,0.75);margin-top:1px;">{nim}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="margin: 1.2rem 0;">
    <div style="font-size:0.6rem;letter-spacing:0.18em;text-transform:uppercase;
                color:#93C4EE;font-weight:600;margin-bottom:0.8rem;">🤖 Informasi Model</div>
    """, unsafe_allow_html=True)

    model_info = [
        ("Model",       "IndoT5-base"),
        ("Task",        "Question Generation"),
        ("Dataset",     "IDK-MRC"),
        ("Pipeline",    "9-stage preprocessing"),
        ("Framework",   "HuggingFace Transformers"),
        ("Device",      DEVICE.upper()),
    ]
    for label, val in model_info:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    margin-bottom:0.45rem;">
            <span style="font-size:0.72rem;color:rgba(147,196,238,0.75);">{label}</span>
            <span style="font-size:0.72rem;font-weight:600;color:#FFFFFF;
                         background:rgba(255,255,255,0.1);border-radius:6px;
                         padding:0.15rem 0.5rem;">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2rem;font-size:0.65rem;color:rgba(147,196,238,0.5);text-align:center;">
        © 2026 · Kelompok AQG
    </div>
    """, unsafe_allow_html=True)

# ── JS: paksa sidebar tetap visible setiap rerun ─────────────────────────────
st.markdown("""
<script>
(function(){
    if(document.getElementById('aqg-sb-fix'))return;
    var s=document.createElement('style');
    s.id='aqg-sb-fix';
    s.textContent=
        '[data-testid="stSidebarCollapseButton"],'
        +'[data-testid="stSidebarOpenButton"],'
        +'[data-testid="collapsedControl"]'
        +'{visibility:visible!important;display:flex!important;opacity:1!important;pointer-events:auto!important;}';
    document.head.appendChild(s);
    function fix(){
        document.querySelectorAll(
            '[data-testid="stSidebarCollapseButton"],'
            +'[data-testid="stSidebarOpenButton"],'
            +'[data-testid="collapsedControl"]'
        ).forEach(function(e){
            e.style.setProperty('visibility','visible','important');
            e.style.setProperty('opacity','1','important');
            e.style.setProperty('display','flex','important');
            e.style.setProperty('pointer-events','auto','important');
        });
    }
    fix();
    new MutationObserver(fix).observe(document.body,{childList:true,subtree:true});
})();
</script>
""", unsafe_allow_html=True)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --blue-900:   #1E3A5F;
    --blue-700:   #2563A8;
    --blue-500:   #4A90D9;
    --blue-300:   #93C4EE;
    --blue-100:   #DBEAFE;
    --blue-50:    #EFF6FF;
    --bg-main:    #F0F6FF;
    --bg-card:    #FFFFFF;
    --bg-muted:   #E8F0FD;
    --text-dark:  #1A2B45;
    --text-mid:   #3D5A80;
    --text-soft:  #6B87A8;
    --border:     #C8DCF5;
    --border-soft:#DDEAF9;
    --green-bg:   #E8F7F2;
    --red-bg:     #FEF2F2;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-dark);
}

#MainMenu { visibility: hidden; }

.main .block-container {
    background: var(--bg-main);
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 720px;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, var(--blue-900) 0%, var(--blue-700) 100%);
    border-radius: 20px;
    padding: 1.8rem 1.6rem 1.4rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:180px; height:180px; background:rgba(255,255,255,0.04); border-radius:50%;
}
.hero::after {
    content:''; position:absolute; bottom:-60px; left:-20px;
    width:220px; height:220px; background:rgba(255,255,255,0.03); border-radius:50%;
}
.hero-eyebrow {
    font-size:0.62rem; letter-spacing:0.22em; text-transform:uppercase;
    color:var(--blue-300); font-weight:600; margin-bottom:0.6rem;
}
.hero-title {
    font-family:'Lora',serif; font-size:1.6rem; line-height:1.18;
    color:#FFF; font-weight:600; margin:0 0 0.4rem;
}
.hero-title em { font-style:italic; color:#93C4EE; font-weight:400; }
.hero-sub { font-size:0.8rem; color:rgba(147,196,238,0.85); font-weight:300; margin-top:0.5rem; }

/* ── SECTION LABEL ── */
.sec-label {
    font-size:0.65rem; letter-spacing:0.16em; text-transform:uppercase;
    color:var(--text-soft); border-left:3px solid var(--blue-500);
    padding-left:0.7rem; margin-bottom:1rem; font-weight:600;
}

/* ── CARD ── */
.card {
    background:var(--bg-card); border:1px solid var(--border-soft);
    border-radius:14px; padding:1.6rem 1.8rem; margin-bottom:1rem;
}

/* ── POPUP ── */
@keyframes popIn {
    from { opacity:0; transform:scale(0.82) translateY(24px); }
    to   { opacity:1; transform:scale(1) translateY(0); }
}

/* ── QUIZ ── */
.quiz-meta {
    display:flex; align-items:center; justify-content:space-between;
    margin-bottom:0.9rem;
}
.quiz-meta-label {
    font-size:0.63rem; letter-spacing:0.16em; text-transform:uppercase;
    color:var(--text-soft); font-weight:600;
}
.quiz-meta-counter {
    font-family:'Lora',serif; font-size:1rem;
    color:var(--blue-700); font-weight:600;
}
.progress-bar-wrap {
    background:var(--blue-100); border-radius:99px;
    height:5px; margin-bottom:1.8rem; overflow:hidden;
}
.progress-bar-fill {
    height:100%;
    background:linear-gradient(90deg, var(--blue-700), var(--blue-500));
    border-radius:99px; transition:width 0.45s ease;
}
.quiz-q-label {
    font-size:0.62rem; letter-spacing:0.16em; text-transform:uppercase;
    color:var(--blue-500); font-weight:600; margin-bottom:0.55rem;
}
.quiz-q-text {
    font-family:'Lora',serif; font-size:1.15rem;
    color:var(--text-dark); line-height:1.58; margin-bottom:1.5rem;
}

/* ── RADIO ── */
/*
   Fix alignment opsi kuis v2:
   Masalahnya bukan panjang teks, tetapi wrapper bawaan Streamlit/BaseWeb
   masih memakai lebar fit-content. Jadi yang dipaksa full width bukan hanya
   label, tetapi juga element-container, radiogroup, dan wrapper radio-nya.
*/

/* Paksa container widget radio memenuhi lebar area utama */
div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]),
div.element-container:has(div[data-testid="stRadio"]),
div[data-testid="stRadio"] {
    width:100% !important;
    max-width:100% !important;
    min-width:100% !important;
    box-sizing:border-box !important;
}

/* Sembunyikan label bawaan "Pilih jawaban:" */
div[data-testid="stRadio"] > label {
    display:none !important;
}

/* Paksa semua wrapper utama radio full width */
div[data-testid="stRadio"] > div,
div[data-testid="stRadio"] [role="radiogroup"],
div[data-testid="stRadio"] [role="radiogroup"] > div,
div[data-testid="stRadio"] [data-baseweb="radio"],
div[data-testid="stRadio"] [data-baseweb="radio"] > div,
div[data-testid="stRadio"] label {
    width:100% !important;
    max-width:100% !important;
    box-sizing:border-box !important;
}

/* Radiogroup dibuat vertikal dan tiap item stretch */
div[data-testid="stRadio"] [role="radiogroup"],
div[data-testid="stRadio"] > div {
    display:flex !important;
    flex-direction:column !important;
    align-items:stretch !important;
    gap:9px !important;
}

/* Kotak putih opsi jawaban */
div[data-testid="stRadio"] [role="radiogroup"] label,
div[data-testid="stRadio"] [data-baseweb="radio"] {
    background:var(--bg-card) !important;
    border:1.5px solid var(--border) !important;
    border-radius:10px !important;
    padding:0.82rem 1.1rem !important;
    cursor:pointer !important;
    transition:border-color 0.15s, background 0.15s !important;
    color:var(--text-dark) !important;
    font-size:0.88rem !important;
    min-height:54px !important;
    display:flex !important;
    align-items:center !important;
    justify-content:flex-start !important;
    box-sizing:border-box !important;
}

/* Hover dan selected */
div[data-testid="stRadio"] [role="radiogroup"] label:hover,
div[data-testid="stRadio"] [data-baseweb="radio"]:hover {
    border-color:var(--blue-500) !important;
    background:var(--blue-50) !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"],
div[data-testid="stRadio"] [data-baseweb="radio"][data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {
    border-color:var(--blue-700) !important;
    background:var(--blue-50) !important;
    color:var(--blue-700) !important;
    font-weight:600 !important;
}

/* Lingkaran radio jangan ikut melebar */
div[data-testid="stRadio"] input,
div[data-testid="stRadio"] [role="radio"],
div[data-testid="stRadio"] label > div:first-child {
    width:auto !important;
    max-width:none !important;
    min-width:auto !important;
    flex:0 0 auto !important;
}

/* Teks jawaban mengisi sisa ruang dan wrap rapi */
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] [data-baseweb="radio"] p,
div[data-testid="stRadio"] [role="radiogroup"] p {
    color:var(--text-dark) !important;
    white-space:normal !important;
    line-height:1.55 !important;
    margin:0 !important;
    flex:1 1 auto !important;
    min-width:0 !important;
}

/* Sembunyikan opsi placeholder kosong */
div[data-testid="stRadio"] [role="radiogroup"] > label:first-child,
div[data-testid="stRadio"] [role="radiogroup"] > div:first-child,
div[data-testid="stRadio"] [data-baseweb="radio"]:first-child {
    display:none !important;
}

/* ── RESULT ── */
.result-card {
    background:var(--bg-card); border:1px solid var(--border-soft);
    border-radius:12px; padding:1.1rem 1.4rem; margin-bottom:0.7rem;
    border-left:4px solid var(--border);
}
.result-card.correct { border-left-color:#16A34A; background:var(--green-bg); }
.result-card.wrong   { border-left-color:#DC2626; background:var(--red-bg); }
.result-q   { font-family:'Lora',serif; font-size:0.93rem; color:var(--text-dark); margin-bottom:0.35rem; }
.result-ans { font-size:0.8rem; color:var(--text-mid); }

/* ── SCORE ── */
.score-wrap { text-align:center; padding:1.8rem 0 1.4rem; }
.score-big  { font-family:'Lora',serif; font-size:4.5rem; font-weight:600; color:var(--blue-700); line-height:1; }
.score-label {
    font-size:0.68rem; letter-spacing:0.16em; text-transform:uppercase;
    color:var(--text-soft); margin-top:0.4rem; font-weight:600;
}
.score-grade {
    display:inline-block; background:var(--blue-50); color:var(--blue-700);
    border:1.5px solid var(--blue-100); border-radius:99px;
    font-size:0.74rem; font-weight:700; padding:0.28rem 1.2rem;
    margin-top:0.7rem; letter-spacing:0.08em; text-transform:uppercase;
}

/* ── BUTTONS ── */
.stButton > button {
    background:var(--blue-700) !important; color:#FFF !important;
    border:none !important; border-radius:10px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:0.87rem !important; font-weight:600 !important;
    padding:0.65rem 1.8rem !important;
    transition:background 0.15s, transform 0.1s !important;
    letter-spacing:0.01em !important;
}
.stButton > button:hover {
    background:var(--blue-900) !important;
    transform:translateY(-1px) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background:var(--bg-card) !important;
    border:2px dashed var(--border) !important;
    border-radius:14px !important;
    transition:border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color:var(--blue-500) !important; }

/* progress bar */
div[data-testid="stProgress"] > div > div { background:var(--blue-500) !important; }

/* slider */
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    background:var(--blue-700) !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

from preprocess import full_preprocess_pipeline as _preprocess_pipeline, pick_answer
from preprocess import _TOC_LINE_RE, _COLOPHON_RE, _PUBLICATION_META_RE
from preprocess import generate_misleading_distractors, _is_clean_distractor

# Wrapper tanpa @st.cache_data agar tidak ada dependency Streamlit di preprocess.py.
# Cache bisa diaktifkan kembali dengan uncomment dekorator jika diperlukan.
# @st.cache_data(show_spinner=False)
def full_preprocess_pipeline(raw_text: str):
    return _preprocess_pipeline(raw_text)


def _is_noise_page(text: str) -> bool:
    """
    Deteksi halaman/slide/section non-konten berdasarkan isi penuh.
    Berlaku untuk PDF (halaman), PPTX (slide), dan DOCX (section).

    Lebih akurat dari filter per-paragraf karena melihat konteks penuh
    satu unit dokumen sekaligus — bukan potongan kecil yang kehilangan konteks.
    """
    if not text or not text.strip():
        return True

    low   = text.lower().strip()
    words = low.split()

    # Terlalu pendek — cover, halaman kosong, slide judul saja
    if len(words) < 15:
        return True

    # Keyword noise di 50 kata pertama → halaman Kata Pengantar / Daftar Isi
    noise_kw = [
        "kata pengantar", "daftar isi", "daftar pustaka",
        "daftar gambar", "daftar tabel", "ucapan terima kasih",
        "prakata", "sanwacana", "persembahan", "daftar singkatan",
        "lembar pengesahan", "halaman judul", "abstrak",
    ]
    first_50 = " ".join(words[:50])
    for kw in noise_kw:
        if kw in first_50:
            return True

    # Pola TOC — mayoritas baris berupa "Judul ... nomor halaman"
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        from preprocess import _TOC_LINE_RE
        toc_count = sum(1 for l in lines if _TOC_LINE_RE.match(l))
        if toc_count / len(lines) > 0.35:
            return True

    # Halaman ucapan terima kasih / prakata — pola khas
    gratitude_kw = [
        "mengucapkan terima kasih", "terima kasih kepada",
        "rasa syukur", "puji syukur", "alhamdulillah",
        "segala puji", "rahmat dan hidayah", "karunia-nya",
        "penyusun mengucapkan", "penulis mengucapkan",
    ]
    if sum(1 for kw in gratitude_kw if kw in low) >= 2:
        return True

    # Identitas buku / cover — nama program studi, mata kuliah, dll
    cover_kw = [
        "program studi", "mata kuliah", "disusun oleh",
        "tim pengampu", "tahun ajaran", "semester genap",
        "semester ganjil", "universitas", "fakultas",
        "nim :", "nim:", "nip :", "nip:",
    ]
    if sum(1 for kw in cover_kw if kw in low) >= 3:
        return True

    # ISBN / copyright / kolofon penerbit
    from preprocess import _PUBLICATION_META_RE
    if _PUBLICATION_META_RE.search(low):
        return True

    return False


def extract_text_from_pdf(b: bytes) -> str:
    """
    Ekstrak teks PDF per halaman, filter halaman noise sebelum digabung.
    Halaman Kata Pengantar, Daftar Isi, cover, dan kolofon dibuang.
    """
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if not t:
                    continue
                # Filter per halaman — lebih akurat dari filter per-paragraf
                if _is_noise_page(t):
                    continue
                parts.append(t)
        return "\n\n".join(parts) if parts else "[Error PDF: semua halaman difilter sebagai noise]"
    except Exception as e:
        return f"[Error PDF: {e}]"


def extract_text_from_docx(b: bytes) -> str:
    """
    Ekstrak teks DOCX per section/heading, filter section noise.
    Section dipisahkan berdasarkan heading (Heading 1/2) sehingga
    Kata Pengantar, Daftar Isi, dll bisa dideteksi sebagai unit utuh.
    """
    try:
        from docx import Document
        from docx.oxml.ns import qn

        doc = Document(io.BytesIO(b))
        sections, current = [], []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style = para.style.name if para.style else ""

            # Heading 1/2 → batas section baru
            if "Heading 1" in style or "Heading 2" in style:
                if current:
                    section_text = "\n".join(current)
                    if not _is_noise_page(section_text):
                        sections.append(section_text)
                current = [text]
            else:
                current.append(text)

        # Section terakhir
        if current:
            section_text = "\n".join(current)
            if not _is_noise_page(section_text):
                sections.append(section_text)

        # Fallback: kalau tidak ada heading sama sekali, filter per paragraf
        if not sections:
            all_paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            # Gabungkan jadi satu teks, filter noise di level seluruh dokumen
            full_text = "\n".join(all_paras)
            if not _is_noise_page(full_text):
                return full_text
            return "[Error DOCX: konten tidak terdeteksi]"

        return "\n\n".join(sections)
    except Exception as e:
        return f"[Error DOCX: {e}]"


def extract_text_from_pptx(b: bytes) -> str:
    """
    Ekstrak teks PPTX per slide, filter slide noise sebelum digabung.
    Slide cover, slide daftar isi, dan slide ucapan terima kasih dibuang.
    Teks dalam slide diurutkan berdasarkan posisi (atas→bawah, kiri→kanan).
    """
    try:
        from pptx import Presentation
        parts = []
        prs   = Presentation(io.BytesIO(b))

        for slide in prs.slides:
            shapes_with_text = []
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text.strip():
                    top  = shape.top  if getattr(shape, "top",  None) is not None else 0
                    left = shape.left if getattr(shape, "left", None) is not None else 0
                    shapes_with_text.append((top, left, shape.text_frame.text.strip()))

            # Urutan baca: atas→bawah, kiri→kanan
            shapes_with_text.sort(key=lambda x: (x[0], x[1]))
            slide_text = "\n".join(s[2] for s in shapes_with_text)

            if not slide_text.strip():
                continue

            # Filter per slide — buang slide noise
            if _is_noise_page(slide_text):
                continue

            parts.append(slide_text)

        return "\n\n".join(parts) if parts else "[Error PPTX: semua slide difilter sebagai noise]"
    except Exception as e:
        return f"[Error PPTX: {e}]"


@st.cache_resource(show_spinner=False)
def load_model():
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    repo_id = "nayii/aqg-menggunakan-indot5"
    try:
        tokenizer = T5Tokenizer.from_pretrained(repo_id)
        model     = T5ForConditionalGeneration.from_pretrained(
            repo_id,
            use_safetensors=True,
            torch_dtype=torch.float32,
        )
        model.config.use_cache = True
        model.to(DEVICE).eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)


_GROQ_MODEL = "qwen/qwen3.6-27b"


def get_groq_client():
    """
    Inisialisasi Groq client dari environment variable atau st.secrets.
    Tidak di-cache dengan @st.cache_resource agar API key selalu dibaca fresh.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        return None, "GROQ_API_KEY tidak ditemukan. Set via environment variable atau .streamlit/secrets.toml"
    return Groq(api_key=api_key), None


def _call_groq(prompt: str, max_tokens: int = 400) -> str | None:
    """
    Helper: kirim prompt ke Groq dan return teks response.
    Return None kalau gagal.
    Qwen3 pakai reasoning_effort='none' agar tidak emit <think>...</think> tags
    dan langsung return JSON — lebih cepat dan tidak perlu di-strip.
    """
    client, err = get_groq_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens,
            reasoning_effort="none",   # non-thinking mode → output langsung
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def _parse_json_safe(raw: str) -> dict | None:
    """Parse JSON dengan fallback strip markdown fence."""
    if raw is None:
        return None
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Hapus thinking tags kalau masih ada
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        return json.loads(cleaned)
    except Exception:
        return None


def generate_answer_with_llm(question: str, anchor: str, context: str) -> str | None:
    """
    Tahap 1 — Generate jawaban benar via Groq Qwen3.

    'anchor' adalah kalimat dari pick_answer() yang dipakai IndoT5 generate question.
    Dipakai sebagai petunjuk topik ke LLM, bukan sebagai jawaban final.

    Prompt dirancang agar:
    - Jawaban faktual berdasarkan konteks dokumen (bukan pengetahuan umum)
    - Maksimal 20 kata → konsisten dengan panjang distractor
    - Tidak mengulang kalimat pertanyaan
    """
    prompt = f"""Kamu adalah asisten akademik yang membaca dokumen dan menjawab pertanyaan secara tepat.

Konteks dari dokumen:
\"\"\"{context[:500]}\"\"\"

Pertanyaan: {question}
Petunjuk topik: {anchor[:150]}

Tugas:
Berikan jawaban yang BENAR dan FAKTUAL berdasarkan konteks di atas.
Syarat:
- Maksimal 20 kata
- Berdasarkan isi konteks, bukan pengetahuan umum
- Berbentuk frasa atau kalimat lengkap
- Tidak mengulang kata-kata dari kalimat pertanyaan secara verbatim

Balas HANYA dengan JSON (tanpa penjelasan, tanpa markdown):
{{"answer": "jawaban di sini"}}"""

    raw  = _call_groq(prompt, max_tokens=150)
    data = _parse_json_safe(raw)
    if data and isinstance(data.get("answer"), str) and data["answer"].strip():
        return data["answer"].strip()
    return None


def generate_distractors_with_llm(question: str, correct: str, context: str) -> list | None:
    """
    Tahap 2 — Generate 3 distractor via Groq Qwen3.

    Dipanggil SETELAH jawaban benar sudah pasti, sehingga distractor bisa
    dikontraskan langsung dengan jawaban benar yang konkret.

    Prompt dirancang agar:
    - Distractor relevan secara semantik (dari domain yang sama)
    - Tidak pakai negasi eksplisit ("bukan", "tidak") — terlalu mudah ditebak
    - Panjang mirip jawaban benar (±5 kata)
    - Tidak terlalu mirip satu sama lain
    """
    prompt = f"""Kamu adalah pembuat soal pilihan ganda Bahasa Indonesia yang ahli.

Konteks dari dokumen:
\"\"\"{context[:500]}\"\"\"

Pertanyaan: {question}
Jawaban benar: {correct}

Tugas:
Buat 3 pilihan jawaban yang SALAH tapi MENGECOH.
Syarat setiap distractor:
- Salah secara faktual berdasarkan konteks
- Terdengar masuk akal dan relevan dengan topik
- Panjang mirip jawaban benar (±5 kata)
- Menggunakan kosakata dari konteks yang sama
- JANGAN gunakan kata "bukan", "tidak", atau negasi eksplisit lainnya
- Ketiga distractor harus berbeda satu sama lain

Balas HANYA dengan JSON (tanpa penjelasan, tanpa markdown):
{{"distractors": ["distractor1", "distractor2", "distractor3"]}}"""

    raw  = _call_groq(prompt, max_tokens=250)
    data = _parse_json_safe(raw)
    if data and isinstance(data.get("distractors"), list):
        result = [d.strip() for d in data["distractors"] if isinstance(d, str) and d.strip()]
        if len(result) >= 1:
            return result[:3]
    return None


def _get_focused_context(chunk_text: str, answer: str, window_sents: int = 3) -> str:
    """
    Ambil kalimat sekitar answer dari chunk agar model fokus ke topik yang benar.
    Masalah sebelumnya: model diberi 200+ kata context, bisa fokus ke topik lain.
    """
    try:
        import nltk as _n
        sents = _n.sent_tokenize(chunk_text)
    except Exception:
        sents = re.split(r'(?<=[.!?])\s+', chunk_text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return chunk_text
    ans_low = answer.lower().strip()[:50]
    ans_idx = 0
    for i, s in enumerate(sents):
        if ans_low in s.lower():
            ans_idx = i
            break
    start = max(0, ans_idx - window_sents)
    end   = min(len(sents), ans_idx + window_sents + 1)
    return ' '.join(sents[start:end])


# Template pertanyaan bervariasi — dipilih berdasarkan tipe kalimat answer
# Mencakup: Apa, Mengapa, Bagaimana, Kapan, Apa yang dimaksud
_Q_TEMPLATES = {
    "definition": [
        "generate question type: definition answer: {answer} context: {context}",
        "generate question: answer: {answer} context: {context}",
        "generate question type: what is answer: {answer} context: {context}",
    ],
    "function": [
        "generate question type: function answer: {answer} context: {context}",
        "generate question type: why answer: {answer} context: {context}",
        "generate question: answer: {answer} context: {context}",
    ],
    "process": [
        "generate question type: process answer: {answer} context: {context}",
        "generate question type: how answer: {answer} context: {context}",
        "generate question: answer: {answer} context: {context}",
    ],
    "general": [
        "generate question: answer: {answer} context: {context}",
        "generate question type: why answer: {answer} context: {context}",
        "generate question type: how answer: {answer} context: {context}",
        "generate question type: when answer: {answer} context: {context}",
    ],
}

# Rotasi template per soal agar variasi pertanyaan lebih beragam
_template_counter = [0]


def _pick_template(answer: str) -> str:
    """Pilih template berdasarkan tipe answer dan rotasi counter."""
    from preprocess import _DEFINITION_RE, _CHARACTERISTIC_RE, _PROCESS_RE
    if _DEFINITION_RE.match(answer):
        ttype = "definition"
    elif _CHARACTERISTIC_RE.match(answer):
        ttype = "function"
    elif _PROCESS_RE.match(answer):
        ttype = "process"
    else:
        ttype = "general"
    templates = _Q_TEMPLATES[ttype]
    idx = _template_counter[0] % len(templates)
    _template_counter[0] += 1
    return templates[idx]


def generate_question(context: str, answer: str, tokenizer, model) -> str:
    """
    Generate pertanyaan dengan focused context dan template bervariasi.
    Focused context = hanya kalimat sekitar answer (bukan seluruh chunk)
    sehingga model tidak nyasar ke topik lain dalam chunk.
    """
    focused    = _get_focused_context(context, answer, window_sents=3)
    template   = _pick_template(answer)
    input_text = template.format(answer=answer.strip(), context=focused.strip())
    try:
        inputs = tokenizer(
            input_text, max_length=MAX_INPUT_LEN,
            truncation=True, padding=False, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **GEN_CONFIG
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return result if result else "[tidak dapat di-generate]"
    except Exception as e:
        return f"[Error: {e}]"



def build_choices(questions: list) -> tuple:
    """
    Bangun pilihan jawaban dengan arsitektur dua tahap via Groq Qwen3:

    Tahap 1 — generate_answer_with_llm()
        LLM generate jawaban benar berdasarkan pertanyaan + konteks dokumen.
        'anchor' (pick_answer output) hanya dipakai sebagai petunjuk topik.

    Tahap 2 — generate_distractors_with_llm()
        LLM generate 3 distractor dengan kontras eksplisit ke jawaban benar.
        Dua tahap terpisah agar distractor bisa dikontraskan dengan jawaban
        konkret, bukan dengan anchor mentah dari teks.

    Fix scoring bug: simpan correct_idx (index posisi jawaban benar di dalam
    opts) bukan string, agar scoring tidak bergantung pada string matching.

    Return:
        choices_all  : list of list[str] — 4 opsi per soal (sudah diacak)
        correct_all  : list[int]         — index jawaban benar di dalam opts
    """
    # Cek API key di awal — tampilkan warning sekali kalau tidak ada
    _, api_err = get_groq_client()
    if api_err:
        st.warning(f"⚠️ {api_err} — menggunakan metode rule-based sebagai fallback.")

    all_answers = list(set(
        q["pseudo_answer"].strip()
        for q in questions
        if _is_clean_distractor(q["pseudo_answer"].strip())
    ))
    fallbacks = [
        "Semua jawaban di atas benar",
        "Tidak ada jawaban yang tepat",
        "Informasi tidak disebutkan",
    ]

    choices_all = []
    correct_all = []   # sekarang list[int] bukan list[str]

    for q in questions:
        anchor     = q["pseudo_answer"].strip()   # dari pick_answer(), bukan jawaban final
        chunk_text = q.get("context", "")
        question   = q.get("question", "")

        correct      = None
        distractors  = None
        llm_success  = False

        # ── Tahap 1: Generate jawaban benar via LLM ──────────────────────
        if api_err is None:
            correct = generate_answer_with_llm(question, anchor, chunk_text)

        # ── Tahap 2: Generate distractor via LLM ─────────────────────────
        if correct is not None:
            distractors = generate_distractors_with_llm(question, correct, chunk_text)
            if distractors is not None and len(distractors) >= 1:
                llm_success = True

        # ── Fallback rule-based kalau LLM gagal di tahap manapun ─────────
        if not llm_success:
            correct     = anchor   # fallback ke anchor asli
            distractors = generate_misleading_distractors(anchor, chunk_text, n=3)

            if len(distractors) < 3:
                pool = [a for a in all_answers
                        if a.lower() != anchor.lower() and a not in distractors]
                random.shuffle(pool)
                distractors += pool[:3 - len(distractors)]

            fb_idx = 0
            while len(distractors) < 3:
                distractors.append(fallbacks[fb_idx % len(fallbacks)])
                fb_idx += 1

        # Pastikan tepat 3 distractor (potong atau pad)
        while len(distractors) < 3:
            distractors.append(fallbacks[len(distractors) % len(fallbacks)])
        distractors = [
            ' '.join(d.split()[:28]) if len(d.split()) > 28 else d
            for d in distractors[:3]
        ]

        # ── Susun 4 opsi + simpan index jawaban benar ────────────────────
        # Dengan menyimpan index, scoring 100% akurat — tidak perlu string matching.
        opts = distractors[:3] + [correct]
        random.shuffle(opts)
        correct_idx = opts.index(correct)   # index 0-3, pasti ada

        choices_all.append(opts)
        correct_all.append(correct_idx)

    return choices_all, correct_all


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════

defaults = {
    "phase":           "upload",
    "questions":       [],
    "quiz_choices":    [],
    "quiz_correct":    [],   # jawaban benar versi LLM (diformulasikan ulang)
    "quiz_idx":        0,
    "quiz_answers":    {},
    "quiz_score":      0,
    "raw_text":        "",
    "file_name":       "",
    "max_q":           8,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── HERO (selalu tampil) ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">✦ Automatic Question Generation · 2026</div>
    <h1 class="hero-title">Dari teks kuliah<br>ke pertanyaan <em>bermakna</em></h1>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PHASE: UPLOAD
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "upload":

    st.markdown('<div class="sec-label">Upload materi kuliah</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop atau klik — PDF, DOCX, atau PPTX",
        type=["pdf", "docx", "pptx"],
        label_visibility="collapsed",
    )

    if uploaded:
        ftype  = uploaded.name.split(".")[-1].lower()
        fbytes = uploaded.read()

        col_q, col_btn = st.columns([3, 1])
        with col_q:
            max_q = st.slider("Jumlah soal", 3, 20, 8)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Generate →", use_container_width=True, type="primary"):
                with st.spinner("📄 Membaca file..."):
                    extractor = {
                        "pdf" : extract_text_from_pdf,
                        "docx": extract_text_from_docx,
                        "pptx": extract_text_from_pptx,
                    }
                    raw_text = extractor[ftype](fbytes)

                if raw_text.startswith("[Error"):
                    st.error(raw_text)
                else:
                    st.session_state.raw_text  = raw_text
                    st.session_state.file_name = uploaded.name
                    st.session_state.max_q     = max_q
                    st.session_state.phase     = "generating"
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PHASE: GENERATING
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "generating":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Sedang memproses materi...")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Memuat model AI..."):
        tokenizer, model, err = load_model()

    if err:
        st.error(f"❌ {err}")
        if st.button("← Kembali"):
            st.session_state.phase = "upload"
            st.rerun()
        st.stop()

    with st.spinner("Menjalankan preprocessing..."):
        res = full_preprocess_pipeline(st.session_state.raw_text)

    from preprocess import _answer_score as _score_fn

    max_q  = st.session_state.max_q
    chunks = [ch for ch in res["chunks"] if ch.get("text", "").strip()]
    pb     = st.progress(0.0, text="Generating soal...")

    # seen_q: cegah pertanyaan duplikat (exact + near-duplicate)
    # seen_ans: cegah answer yang sama dipakai di soal berbeda
    results, seen_q, seen_ans = [], set(), set()

    def _normalize_q(q: str) -> str:
        """Normalize pertanyaan untuk deteksi duplikat."""
        q = q.lower().strip()
        # Hapus variasi phrasing
        q = re.sub(r'\b(apakah|apakan)\b', 'apa', q)
        q = re.sub(r'\b(bagaimanakah)\b', 'bagaimana', q)
        q = re.sub(r'\b(mengapakah)\b', 'mengapa', q)
        q = re.sub(r'\b(siapakah)\b', 'siapa', q)
        q = re.sub(r'\b(dimanakah|di\s+manakah)\b', 'dimana', q)
        q = re.sub(r'\b(kapankah)\b', 'kapan', q)
        q = re.sub(r'\b(berapakah)\b', 'berapa', q)
        # Normalisasi spasi dalam kata majemuk — fix "di maksud" vs "dimaksud"
        q = re.sub(r'\bdi\s+maksud\b', 'dimaksud', q)
        q = re.sub(r'\bdi\s+mana\b', 'dimana', q)
        q = re.sub(r'\bdi\s+sini\b', 'disini', q)
        q = re.sub(r'\bdi\s+sana\b', 'disana', q)
        q = re.sub(r'\s+', ' ', q).strip()
        return q

    for idx, ch in enumerate(chunks):
        if len(results) >= max_q:
            break
        text = ch.get("text", "").strip()
        if not text:
            continue

        answer = pick_answer(text)
        if not answer or not answer.strip():
            continue

        # Potong answer terlalu panjang
        answer_words = answer.split()
        if len(answer_words) > 35:
            answer = ' '.join(answer_words[:25])

        # Skip kalau answer tidak informatif
        if _score_fn(answer) < 1.5:
            continue

        # Skip kalau answer sudah dipakai (cegah soal kembar)
        ans_key = _normalize_q(answer[:60])
        if ans_key in seen_ans:
            continue

        if not _is_clean_distractor(answer):
            continue

        q = generate_question(text, answer, tokenizer, model)

        q_clean = re.sub(r'\s+([?.!,])', r'\1', q).strip()
        if q_clean:
            q_clean = q_clean[0].upper() + q_clean[1:]

        q_norm = _normalize_q(q_clean)

        # Cek duplikat dengan normalized form (tangkap variasi "apa/apakah")
        is_dup = q_norm in seen_q or len(q_norm) < 8 or q_norm.startswith("[error")

        if not is_dup:
            seen_q.add(q_norm)
            seen_ans.add(ans_key)
            results.append({
                "chunk_id":      ch.get("chunk_id", idx),
                "question":      q_clean,
                "context":       text,
                "pseudo_answer": answer,
            })
        pb.progress(
            min(len(results) / max_q, 1.0),
            text=f"Generating soal... {len(results)}/{max_q}"
        )

    pb.empty()
    choices_all, correct_all          = build_choices(results)
    st.session_state.questions        = results
    st.session_state.quiz_choices     = choices_all
    st.session_state.quiz_correct     = correct_all
    st.session_state.quiz_idx         = 0
    st.session_state.quiz_answers     = {}
    st.session_state.quiz_score       = 0
    st.session_state.phase            = "popup"
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PHASE: POPUP
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "popup":
    n = len(st.session_state.questions)
    max_requested  = st.session_state.max_q
    chunk_limited  = n < max_requested

    badge_html = f"""
        <div style="display:inline-flex;align-items:center;gap:0.4rem;
                    background:#EFF6FF;color:#2563A8;
                    border:1.5px solid #DBEAFE;border-radius:99px;
                    font-size:0.74rem;font-weight:700;padding:0.3rem 1.1rem;
                    margin-bottom:{'0.6rem' if chunk_limited else '1.8rem'};
                    letter-spacing:0.04em;">
            ✦ &nbsp;{n} soal pilihan ganda berhasil di-generate
        </div>
    """
    if chunk_limited:
        badge_html += f"""
        <div style="font-size:0.78rem;color:#B45309;background:#FFFBEB;
                    border:1px solid #FDE68A;border-radius:10px;
                    padding:0.55rem 1rem;margin-bottom:1.8rem;line-height:1.65;
                    text-align:left;">
            ⚠️ Kamu meminta <b>{max_requested} soal</b>, namun chunk materi yang
            tersedia hanya cukup untuk menghasilkan <b>{n} soal</b>.
            Coba upload materi yang lebih panjang untuk mendapatkan lebih banyak soal.
        </div>
        """

    st.markdown("""
    <style>
    .main .block-container { background: rgba(10,20,40,0.55) !important; }
    </style>
    """, unsafe_allow_html=True)

    _, col_mid, _ = st.columns([1, 3, 1])
    with col_mid:
        st.markdown(f"""
        <div style="
            background:#FFF;
            border-radius:22px;
            padding:2.6rem 2.4rem 1.6rem;
            text-align:center;
            box-shadow:0 28px 70px rgba(30,58,95,0.28);
            border-top:4px solid #4A90D9;
            margin-top:2rem;
            animation:popIn 0.38s cubic-bezier(0.34,1.56,0.64,1);
        ">
            <div style="font-size:3rem;margin-bottom:0.9rem;">🎉</div>
            <div style="font-family:'Lora',serif;font-size:1.6rem;font-weight:600;
                        color:#1A2B45;margin-bottom:0.5rem;">Soal siap!</div>
            <p style="font-size:0.84rem;color:#6B87A8;line-height:1.65;margin-bottom:1.4rem;">
                Materi dari <b style="color:#1A2B45;">{st.session_state.file_name}</b><br>
                berhasil diproses menjadi soal pilihan ganda.
            </p>
            {badge_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        if st.button("🚀  Mulai Kuis", use_container_width=True, type="primary"):
            st.session_state.phase = "quiz"
            st.rerun()

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        if st.button("↩  Upload ulang", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PHASE: QUIZ — satu per satu
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "quiz":
    questions   = st.session_state.questions
    choices_all = st.session_state.quiz_choices
    correct_all = st.session_state.quiz_correct   # list[int] — index jawaban benar
    idx         = st.session_state.quiz_idx
    total       = len(questions)

    if idx >= total:
        st.session_state.phase = "result"
        st.rerun()

    q           = questions[idx]
    opts        = choices_all[idx]
    correct_idx = correct_all[idx]          # int: posisi jawaban benar di opts
    correct_str = opts[correct_idx]         # string jawaban benar untuk ditampilkan
    pct         = idx / total * 100

    st.markdown(f"""
    <div class="quiz-meta">
        <span class="quiz-meta-label">Progress kuis</span>
        <span class="quiz-meta-counter">{idx + 1} / {total}</span>
    </div>
    <div class="progress-bar-wrap">
        <div class="progress-bar-fill" style="width:{pct:.1f}%"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <div class="quiz-q-label">Pertanyaan {idx + 1:02d}</div>
        <div class="quiz-q-text">{q['question']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Ambil pilihan sebelumnya kalau user balik ke soal ini
    prev = st.session_state.quiz_answers.get(idx, {})
    prev_selected = prev.get("selected") if isinstance(prev, dict) else None

    _placeholder = ""               # sentinel string kosong — disembunyikan via CSS
    radio_opts   = [_placeholder] + opts   # placeholder di index 0

    if prev_selected and prev_selected in opts:
        default_idx = radio_opts.index(prev_selected)
    else:
        default_idx = 0             # default ke placeholder (tidak ada yang terpilih)

    raw_selected = st.radio(
        "Pilih jawaban:",
        options=radio_opts,
        index=default_idx,
        key=f"quiz_radio_{idx}",
        label_visibility="collapsed",
        format_func=lambda x: x if x else "— Pilih jawaban —",
    )
    selected = raw_selected if raw_selected else None

    st.markdown("<br>", unsafe_allow_html=True)

    col_back, col_spacer, col_next = st.columns([1, 2, 1])
    with col_back:
        if idx > 0:
            if st.button("← Kembali", use_container_width=True):
                st.session_state.quiz_idx = idx - 1
                st.rerun()
    with col_next:
        btn_label = "Selesai ✓" if idx == total - 1 else "Lanjut →"
        if st.button(btn_label, use_container_width=True, type="primary"):
            if not selected:
                st.warning("⚠️ Pilih salah satu jawaban dulu sebelum lanjut.")
            else:
                # Scoring via index — 100% akurat, tidak perlu string matching
                selected_idx = opts.index(selected)
                is_correct   = (selected_idx == correct_idx)
                st.session_state.quiz_answers[idx] = {
                    "selected": selected,
                    "correct":  correct_str,
                    "is_correct": is_correct,
                }
                if is_correct:
                    st.session_state.quiz_score += 1
                st.session_state.quiz_idx = idx + 1
                if st.session_state.quiz_idx >= total:
                    st.session_state.phase = "result"
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PHASE: RESULT
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "result":
    questions   = st.session_state.questions
    answers     = st.session_state.quiz_answers
    correct_all = st.session_state.quiz_correct
    score       = st.session_state.quiz_score
    total       = len(questions)
    pct         = round(score / total * 100) if total else 0
    grade       = "A" if pct >= 85 else "B" if pct >= 70 else "C" if pct >= 55 else "D"

    st.markdown(f"""
    <div class="score-wrap">
        <div class="score-big">{pct}%</div>
        <div class="score-label">Skor akhir · {score} dari {total} benar</div>
        <div class="score-grade">Grade {grade}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Rekap jawaban</div>', unsafe_allow_html=True)

    for i, q in enumerate(questions):
        ans_data = answers.get(i, {})
        usr      = ans_data.get("selected", "(tidak dijawab)") if isinstance(ans_data, dict) else "(tidak dijawab)"
        correct  = ans_data.get("correct", "-") if isinstance(ans_data, dict) else "-"
        is_ok    = ans_data.get("is_correct", False) if isinstance(ans_data, dict) else False

        cls   = "correct" if is_ok else "wrong"
        icon  = "✓" if is_ok else "✕"
        color = "#0F7B55" if is_ok else "#B91C1C"
        st.markdown(f"""
        <div class="result-card {cls}">
            <div style="font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;
                        color:{color};font-weight:700;margin-bottom:0.35rem;">{icon} Soal {i+1:02d}</div>
            <div class="result-q">{q['question']}</div>
            <div class="result-ans">Jawabanmu: <b>{usr}</b></div>
            <div class="result-ans" style="margin-top:0.2rem;">
                Jawaban benar: <b style="color:{color};">{correct}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Ulangi Kuis", use_container_width=True):
            st.session_state.quiz_idx     = 0
            st.session_state.quiz_answers = {}
            st.session_state.quiz_score   = 0
            choices_all, correct_all      = build_choices(questions)
            st.session_state.quiz_choices = choices_all
            st.session_state.quiz_correct = correct_all
            st.session_state.phase        = "quiz"
            st.rerun()
    with col2:
        if st.button("📤 Upload Materi Baru", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()