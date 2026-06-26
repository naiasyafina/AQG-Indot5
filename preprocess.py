"""
preprocess.py — AQG Preprocessing Pipeline
Pure Python logic, tanpa import Streamlit.

Dipakai oleh app.py untuk mengubah teks mentah hasil parsing PDF/PPTX/DOCX
menjadi chunk yang bersih, fokus, dan siap diberikan ke model IndoT5-QTYPE.

Pembagian tanggung jawab:
- Notebook Kaggle : training, evaluasi, upload model ke HuggingFace.
- preprocess.py   : cleaning, noise removal, chunking, pick_answer, question_type.
- app.py          : UI Streamlit, upload file, panggil model, buat quiz.

Pipeline utama:
1.  Cleaning awal (unicode, control char, dehyphenation, spasi).
2.  Filter halaman/slide/section non-konten (cover, TOC, daftar pustaka, dll).
3.  Hapus blok tugas/latihan/pertanyaan.
4.  Rekonstruksi heading, bullet, numbered list menjadi kalimat naratif.
5.  Normalisasi singkatan informal Bahasa Indonesia.
6.  Ekspansi akronim kapital secara aman (inline definition + Wikipedia optional).
7.  Restorasi tanda baca.
8.  Heading-aware + semantic chunking.
9.  Pick answer berbasis scoring dari chunk.
10. Infer question_type untuk model QTYPE.
11. Focused context sekitar answer.
12. Validator heuristic dan distractor heuristic sebagai fallback LLM.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nltk
import requests

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# 0. NLTK SETUP
# ═════════════════════════════════════════════════════════════════════════════

for _pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# 1. REGEX DAN KONSTANTA GLOBAL
# ═════════════════════════════════════════════════════════════════════════════

CTRL_CHAR_RE    = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE  = re.compile(r" {2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
DEHYPHEN_RE     = re.compile(r"(\w)-\n(\w)")
PUNCT_AFTER_RE  = re.compile(r"([.,;:!?])(?=[^\s\d])")
PUNCT_BEFORE_RE = re.compile(r"\s+([.,;:!?])")
SENT_SPLIT_RE   = re.compile(r"(?<=[.!?])\s+")
PARA_SPLIT_RE   = re.compile(r"\n{2,}")

BULLET_RE       = re.compile(r"^\s*[•\-\*\–\→●○►]\s+")
BULLET_CAP_RE   = re.compile(r"^\s*[•\-\*\–\→●○►]\s+(.*)")
NUMBER_RE       = re.compile(r"^\s*(\d+\.|[a-zA-Z]\.)\s+")
NUMBER_CAP_RE   = re.compile(r"^\s*(\d+)\.\s+(.*)")
COMPLETE_RE     = re.compile(r"[.!?]\s*$")
STRIP_NUM_RE    = re.compile(r"^\s*(\d+\.|[a-zA-Z]\.)\s*")

# Baris TOC seperti "BAB I .... 12" atau "Pengertian     4"
_TOC_LINE_RE = re.compile(
    r"^.{3,80}\s+\.{2,}\s*\d{1,4}\s*$"
    r"|^.{3,80}\s{2,}\d{1,4}\s*$"
    r"|^\s*[A-Z]\.\s+.{4,80}\s+\d{1,4}\s*$",
    re.IGNORECASE,
)

_STRUCTURAL_HEADER_RE = re.compile(
    r"^(kata\s+pengantar|daftar\s+isi|daftar\s+pustaka|daftar\s+gambar|"
    r"daftar\s+tabel|daftar\s+singkatan|prakata|sanwacana|persembahan|"
    r"halaman\s+judul|lembar\s+pengesahan|abstrak|abstract|"
    r"biodata\s+penulis|tentang\s+penulis|tentang\s+penyusun|"
    r"profil\s+penulis|riwayat\s+hidup|latihan\s+soal|soal\s+latihan|"
    r"evaluasi|kuis|quiz)\s*$",
    re.IGNORECASE,
)

_PUBLICATION_META_RE = re.compile(
    r"isbn|hak\s+cipta|copyright|©|diterbitkan\s+oleh|penerbit|"
    r"cetakan\s+(ke|pertama|kedua)|all\s+rights\s+reserved|"
    r"editor\s*:|desain\s+sampul|layout\s*:|percetakan",
    re.IGNORECASE,
)

_COLOPHON_RE = re.compile(r"^[A-Z][a-zA-Z\s]+,\s*(19|20)\d{2}\s*$")

# Noise running header/footer khas buku ajar
_RUNNING_HEADER_HINT_RE = re.compile(
    r"\b(bahan\s+ajar|modul|aplikasi\s+komputer|frizka\s+fitriana|"
    r"universitas|fakultas|program\s+studi)\b",
    re.IGNORECASE,
)

_TABLE_NOISE_HINT_RE = re.compile(
    r"\b(sub\s*menu|tombol|fungsinya|keterangan\s*[a-z]?$|sebagai\s+berikut\s*:?|"
    r"ctrl\s*\+|alt\s*\+|toolbar|menu\s+bar)\b",
    re.IGNORECASE,
)

# Noise hasil Wikipedia / OCR
_WIKI_IMAGE_MARKUP_RE = re.compile(
    r"\b(?:jmpl|thumb|thumbnail)\|(?:ka|ki|left|right|pus|center)?\|?\d{2,4}px\|?",
    re.IGNORECASE,
)
_WIKI_TEMPLATE_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)
_WIKI_REF_RE      = re.compile(r"\[\d+\]|\[citation needed\]", re.IGNORECASE)
_WIKI_ERROR_RE    = re.compile(r"error:\s*\{\{.*?\}\}.*?\(help\)", re.IGNORECASE)

CONJUNCTIONS = {
    "dan", "atau", "serta", "yang", "untuk", "dengan", "pada", "ke", "dari",
    "oleh", "dalam", "ini", "itu", "juga", "namun", "karena", "sehingga",
}

TASK_KEYWORDS = {
    "tugas", "latihan", "soal", "buatlah", "kerjakan", "diskusikan",
    "presentasikan", "jawablah", "pilihlah", "evaluasi", "kuis", "quiz",
}

ORDINALS = {
    "1": "Pertama", "2": "Kedua", "3": "Ketiga", "4": "Keempat",
    "5": "Kelima", "6": "Keenam", "7": "Ketujuh", "8": "Kedelapan",
    "9": "Kesembilan",
}


# ═════════════════════════════════════════════════════════════════════════════
# 2. REMOVE REPEATED HEADER/FOOTER/WATERMARK TEXT
# ═════════════════════════════════════════════════════════════════════════════

def normalize_repeated_line(line: str) -> str:
    """
    Normalisasi baris untuk deteksi header/footer/watermark berulang.

    Sengaja lebih kuat untuk buku/PDF yang header-footernya berubah
    karena nomor halaman, misalnya:
    - "2 Bahan Ajar Aplikasi Komputer"
    - "Frizka Fitriana, S.Kom., M.Kom. 9"

    Semua contoh tersebut dinormalisasi menjadi inti teks yang sama,
    sehingga bisa terdeteksi sebagai repeated footer/header lintas halaman.
    """
    line = unicodedata.normalize("NFKC", line or "")
    line = line.strip().lower()
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"[\u00ad\ufffd]", "", line)
    line = re.sub(r"^[\-–—|•\s]+|[\-–—|•\s]+$", "", line)

    # Hilangkan penanda halaman eksplisit
    line = re.sub(r"\b(page|halaman|slide|hlm\.?)\s*[:.-]?\s*\d+\b", "", line, flags=re.IGNORECASE)

    # Hilangkan nomor halaman di awal/akhir baris
    # Penting untuk pola: "2 Bahan Ajar Aplikasi Komputer" dan "Nama Penulis 3"
    line = re.sub(r"^\s*(?:[ivxlcdm]+|\d{1,4})\s+", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\s+(?:[ivxlcdm]+|\d{1,4})\s*$", "", line, flags=re.IGNORECASE)

    # Hilangkan bentuk "3 / 104" atau angka saja
    line = re.sub(r"^\d+\s*/\s*\d+$", "", line)
    line = re.sub(r"^\d+$", "", line)

    # Bersihkan separator umum footer/header
    line = re.sub(r"\s*[|•·]+\s*", " ", line)
    line = re.sub(r"\s+", " ", line).strip(" -–—|•.")
    return line.strip()


def remove_repeated_headers_footers(
    page_texts: Sequence[str],
    min_ratio: float = 0.30,
    max_line_chars: int = 110,
    top_bottom_window: int = 7,
) -> List[str]:
    """
    Menghapus baris berulang di banyak halaman/slide yang merupakan
    header/footer/watermark.

    Parameter:
    - min_ratio: proporsi minimum halaman yang harus mengandung baris agar
      dianggap repeated (default 30%).
    - top_bottom_window: berapa baris teratas/terbawah yang dicek per halaman.

    Urutan:
    1. Ambil beberapa baris bagian atas dan bawah halaman.
    2. Normalisasi baris, termasuk menghapus nomor halaman.
    3. Hitung kemunculan lintas halaman.
    4. Buang baris repeated dari seluruh halaman.
    5. Buang juga baris running header/footer yang jelas walau tidak identik.
    """
    if not page_texts:
        return []

    counter: Counter[str] = Counter()
    page_line_norms: List[List[str]] = []

    for page in page_texts:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        norms_for_page: List[str] = []
        if lines:
            candidate_lines = lines[:top_bottom_window] + lines[-top_bottom_window:]
            for line in candidate_lines:
                norm = normalize_repeated_line(line)
                if norm and len(norm) <= max_line_chars:
                    norms_for_page.append(norm)
            counter.update(set(norms_for_page))
        page_line_norms.append(norms_for_page)

    min_count = max(2, math.ceil(len(page_texts) * min_ratio))
    repeated = {line for line, count in counter.items() if count >= min_count}

    cleaned_pages: List[str] = []
    for page in page_texts:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        cleaned: List[str] = []
        for idx, line in enumerate(lines):
            norm = normalize_repeated_line(line)
            is_top_or_bottom = (idx < top_bottom_window or idx >= len(lines) - top_bottom_window)

            # Case 1: repeated header/footer lintas halaman
            if norm in repeated:
                continue

            # Case 2: running header/footer jelas di area atas/bawah
            # Contoh: "12 Bahan Ajar Aplikasi Komputer" atau "Frizka ... 13"
            if is_top_or_bottom and _RUNNING_HEADER_HINT_RE.search(norm) and len(norm.split()) <= 8:
                continue

            cleaned.append(line)
        cleaned_pages.append("\n".join(cleaned).strip())

    return cleaned_pages


# ═════════════════════════════════════════════════════════════════════════════
# 3. FILTER HALAMAN/SLIDE/SECTION NON-KONTEN
# ═════════════════════════════════════════════════════════════════════════════

def is_toc_block(text: str) -> bool:
    """Deteksi blok daftar isi berdasarkan pola baris TOC."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False

    first = lines[0].lower()
    if "daftar isi" in first:
        return True

    toc_lines = sum(1 for l in lines if _TOC_LINE_RE.match(l))
    return len(lines) >= 4 and toc_lines / max(1, len(lines)) >= 0.35


def is_noise_unit(text: str) -> bool:
    """
    Deteksi halaman/slide/section non-konten.

    Unit = satu halaman PDF, satu slide PPTX, atau satu section DOCX.
    Tidak terlalu agresif: jika unit masih punya banyak kalimat edukatif,
    jangan dibuang walaupun ada kata "universitas".
    """
    if not text or not text.strip():
        return True

    raw = text.strip()
    low = raw.lower()
    words = low.split()

    if len(words) < 12:
        return True

    first_60 = " ".join(words[:60])

    early_noise_kw = [
        "kata pengantar", "daftar isi", "daftar pustaka", "daftar gambar",
        "daftar tabel", "ucapan terima kasih", "prakata", "sanwacana",
        "persembahan", "lembar pengesahan", "halaman judul", "cover",
        "latihan soal", "soal latihan",
    ]
    if any(kw in first_60 for kw in early_noise_kw):
        return True

    if is_toc_block(raw):
        return True

    if _PUBLICATION_META_RE.search(low):
        if len(words) < 120 or sum(
            1 for x in ["isbn", "penerbit", "copyright", "hak cipta"] if x in low
        ) >= 2:
            return True

    task_count = sum(1 for kw in TASK_KEYWORDS if kw in low)
    if task_count >= 3 and len(words) < 160:
        return True

    cover_kw = ["program studi", "fakultas", "universitas", "disusun oleh", "mata kuliah", "nim", "nip"]
    if sum(1 for kw in cover_kw if kw in first_60) >= 3:
        return True

    return False


def is_educational_paragraph(text: str) -> bool:
    """
    Filter paragraf non-edukatif setelah raw text digabung.

    Menolak:
    - metadata / TOC
    - blok tugas/latihan/pertanyaan
    - running header/footer
    - potongan tabel yang tidak informatif
    """
    if not text or not text.strip():
        return False

    stripped = text.strip()
    low = stripped.lower()
    words = low.split()

    if len(words) < 4:
        return False

    norm_line = normalize_repeated_line(stripped)
    if _RUNNING_HEADER_HINT_RE.search(norm_line) and len(norm_line.split()) <= 8:
        return False

    if _STRUCTURAL_HEADER_RE.match(stripped):
        return False
    if _COLOPHON_RE.match(stripped):
        return False
    if _TOC_LINE_RE.match(stripped):
        return False
    if _PUBLICATION_META_RE.search(low):
        return False
    if is_toc_block(stripped):
        return False

    first = words[0] if words else ""
    if first in TASK_KEYWORDS:
        return False

    # Potongan tabel menu/toolbar biasanya buruk untuk AQG
    table_hits = len(_TABLE_NOISE_HINT_RE.findall(low))
    if table_hits >= 2 and len(words) < 80:
        return False

    return True


def remove_task_blocks(paras: Sequence[str]) -> List[str]:
    """
    Buang blok tugas/latihan/pertanyaan/evaluasi.

    Ketika menemukan heading "Tugas" atau "Latihan", sistem membuang
    paragraf setelahnya sampai menemukan heading bab/section baru.
    Ini mencegah pertanyaan bawaan buku ikut dijadikan bahan soal.
    """
    result: List[str] = []
    skipping = False

    stop_heading_re = re.compile(
        r"^(bab\s+[ivxlcdm0-9]+|chapter\s+\d+|bagian\s+\d+|[A-Z][A-Za-zÀ-ÿ0-9 /-]{2,70})$",
        re.IGNORECASE,
    )
    task_heading_re = re.compile(
        r"^(tugas|latihan|pertanyaan|soal|evaluasi|kuis|quiz)(\s+.*)?$",
        re.IGNORECASE,
    )

    for p in paras:
        stripped = p.strip()
        low = stripped.lower()
        first_line = stripped.splitlines()[0].strip() if stripped.splitlines() else stripped

        if task_heading_re.match(first_line):
            skipping = True
            continue

        if skipping:
            # Mulai ambil lagi kalau bertemu BAB/heading materi baru
            if (stop_heading_re.match(first_line)
                    and not NUMBER_RE.match(first_line)
                    and len(first_line.split()) <= 8):
                skipping = False
            else:
                continue

        # Buang paragraf instruksi walau tidak didahului heading tugas
        task_hits = sum(1 for kw in TASK_KEYWORDS if kw in low)
        if task_hits >= 2 and any(
            x in low for x in ["buat", "kerjakan", "presentasi", "kelompok", "jawab"]
        ):
            continue

        result.append(p)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 4. CLEANING DASAR
# ═════════════════════════════════════════════════════════════════════════════

_INFORMAL_ABBR = {
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "krn": "karena",
    "krna": "karena", "utk": "untuk", "dlm": "dalam", "thd": "terhadap",
    "thdp": "terhadap", "pd": "pada", "dr": "dari", "sbg": "sebagai",
    "spt": "seperti", "spy": "supaya", "stlh": "setelah", "sblm": "sebelum",
    "sdh": "sudah", "blm": "belum", "jg": "juga", "lg": "lagi",
    "lbh": "lebih", "krg": "kurang", "kpd": "kepada", "tsb": "tersebut",
    "dpt": "dapat", "bs": "bisa", "ttg": "tentang", "mjd": "menjadi",
    "adl": "adalah", "tdk": "tidak", "hrs": "harus", "scr": "secara",
    "dll": "dan lain-lain", "dsb": "dan sebagainya", "dst": "dan seterusnya",
    "dkk": "dan kawan-kawan",
}
_INFORMAL_ABBR_PATTERNS = {
    k: re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE)
    for k in _INFORMAL_ABBR
}

ABBR_DETECT_RE  = re.compile(r"\b([A-Z][A-Z/]{1,6})\b")
WIKIPEDIA_API   = "https://id.wikipedia.org/w/api.php"
MAX_WIKI_LOOKUPS = 15
WIKIPEDIA_TIMEOUT = 3


def normalize_informal_abbreviations(text: str) -> str:
    """Normalisasi singkatan informal Bahasa Indonesia.
    
    Contoh: yg → yang, dgn → dengan, adl → adalah, dll → dan lain-lain
    """
    for abbr, expansion in _INFORMAL_ABBR.items():
        text = _INFORMAL_ABBR_PATTERNS[abbr].sub(expansion, text)
    return text


def clean_wiki_noise(text: str) -> str:
    """Bersihkan noise khas dataset Wikipedia / OCR."""
    text = _WIKI_ERROR_RE.sub(" ", text)
    text = _WIKI_TEMPLATE_RE.sub(" ", text)
    text = _WIKI_IMAGE_MARKUP_RE.sub(" ", text)
    text = _WIKI_REF_RE.sub(" ", text)
    return text


def clean_text(text: str) -> str:
    """
    Tahap 1 — cleaning awal:
    - unicode normalization (NFKC)
    - hapus control character
    - hapus wiki/image markup noise
    - dehyphenation PDF (kata terpotong di akhir baris)
    - rapikan spasi dan baris kosong berlebih
    - rapikan punctuation yang menempel langsung di huruf
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = CTRL_CHAR_RE.sub(" ", text)
    text = clean_wiki_noise(text)
    text = DEHYPHEN_RE.sub(r"\1\2", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\.{4,}", " ", text)

    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            lines.append("")
            continue
        # Jangan pecah angka 1.173,77; hanya rapikan punctuation yang menempel huruf
        line = PUNCT_AFTER_RE.sub(r"\1 ", line)
        line = PUNCT_BEFORE_RE.sub(r"\1", line)
        line = MULTI_SPACE_RE.sub(" ", line)
        lines.append(line)

    text = "\n".join(lines)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# 5. AKRONIM KAPITAL
# ═════════════════════════════════════════════════════════════════════════════

def extract_inline_definition(text: str, abbr: str) -> Optional[str]:
    """Cari definisi akronim yang sudah ada di teks (inline)."""
    before = re.compile(r"((?:[A-Z][A-Za-z]+\s+){1,6})\(" + re.escape(abbr) + r"\)")
    after  = re.compile(re.escape(abbr) + r"\s*\(((?:[A-Z][A-Za-z]+\s*){1,6})\)")

    m = before.search(text)
    if m and len(m.group(1).split()) >= 2:
        return f"{m.group(1).strip()} ({abbr})"

    m = after.search(text)
    if m and len(m.group(1).split()) >= 2:
        return f"{m.group(1).strip()} ({abbr})"

    return None


@lru_cache(maxsize=512)
def lookup_wikipedia_abbr(abbr: str) -> Optional[str]:
    """
    Lookup Wikipedia Bahasa Indonesia untuk akronim kapital.
    Jika gagal, return None — lebih baik tidak expand daripada expand salah.
    Ada timeout dan circuit breaker agar tidak memperlambat pipeline.
    """
    try:
        resp = requests.get(
            WIKIPEDIA_API,
            params={
                "action": "opensearch",
                "search": abbr,
                "limit": 3,
                "namespace": 0,
                "format": "json",
            },
            timeout=WIKIPEDIA_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        titles = data[1] if len(data) > 1 else []
        chars = [c for c in abbr if c.isalpha()]

        for title in titles:
            title_upper = title.upper()
            ratio = sum(1 for c in chars if c in title_upper) / max(1, len(chars))
            if ratio >= 0.75 and len(title.split()) >= 2:
                return f"{title} ({abbr})"
    except Exception:
        return None
    return None


def expand_abbreviations(text: str) -> str:
    """
    Ekspansi akronim kapital (CPU, RAM, OS, dll).

    Strategi:
    1. Cari definisi inline yang sudah ada di teks, misal "Central Processing Unit (CPU)".
    2. Wikipedia lookup dengan circuit breaker (MAX_WIKI_LOOKUPS).
    3. Skip jika tidak yakin — jangan expand kalau ragu.
    """
    candidates = list(dict.fromkeys(ABBR_DETECT_RE.findall(text)))
    lookups = 0
    expanded: set[str] = set()

    for abbr in candidates:
        if abbr in expanded or len(abbr) < 2:
            continue

        expansion = extract_inline_definition(text, abbr)

        if expansion is None and lookups < MAX_WIKI_LOOKUPS:
            expansion = lookup_wikipedia_abbr(abbr)
            lookups += 1

        if not expansion:
            continue

        # Jika expansion sudah ada di sekitar abbr, jangan duplikasi
        if expansion.lower() in text.lower():
            expanded.add(abbr)
            continue

        text, n = re.subn(r"\b" + re.escape(abbr) + r"\b", expansion, text, count=1)
        if n:
            expanded.add(abbr)

    return text


# ═════════════════════════════════════════════════════════════════════════════
# 6. REKONSTRUKSI HEADING / BULLET / NUMBERED LIST
# ═════════════════════════════════════════════════════════════════════════════

def looks_like_heading(text: str) -> bool:
    """Deteksi heading pendek (≤10 kata, tidak diakhiri tanda baca kalimat)."""
    t = text.strip()
    if not t or COMPLETE_RE.search(t):
        return False
    words = t.split()
    if len(words) > 10:
        return False
    if len(words) <= 2:
        return True
    cap_words = sum(1 for w in words if w[:1].isupper() or w.isupper())
    return cap_words / max(1, len(words)) >= 0.55


def heading_to_sentence(text: str) -> str:
    """Heading pendek dijadikan kalimat pengantar agar konteksnya tidak hilang."""
    t = STRIP_NUM_RE.sub("", text.strip()).strip()
    if not t:
        return text
    return f"Bagian ini membahas tentang {t}."


def reconstruct_bullets(text: str) -> str:
    """
    Ubah bullet list menjadi kalimat naratif.

    Contoh:
    Tujuan normalisasi:
    - mengurangi duplikasi data
    - menjaga konsistensi data
    - mencegah anomali

    → "Tujuan normalisasi meliputi mengurangi duplikasi data,
       menjaga konsistensi data, dan mencegah anomali."
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    prefix_parts: List[str] = []
    items: List[str] = []

    for line in lines:
        m = BULLET_CAP_RE.match(line)
        if m:
            items.append(m.group(1).strip().rstrip(".;"))
        else:
            prefix_parts.append(line.strip())

    if not items:
        return text

    if len(items) == 1:
        joined = items[0]
    elif len(items) == 2:
        joined = f"{items[0]} dan {items[1]}"
    else:
        joined = ", ".join(items[:-1]) + f", dan {items[-1]}"

    prefix = " ".join(prefix_parts).strip()
    if prefix.endswith(":"):
        prefix = prefix[:-1]
    if prefix:
        return f"{prefix} meliputi {joined}."
    return f"Terdapat beberapa poin, yaitu {joined}."


def reconstruct_numbered(text: str) -> str:
    """Ubah numbered list menjadi kalimat naratif dengan ordinal Bahasa Indonesia."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    result: List[str] = []
    current_num = ""
    current_text = ""

    for line in lines:
        m = NUMBER_CAP_RE.match(line)
        if m:
            if current_text:
                ord_word = ORDINALS.get(current_num, f"Poin {current_num}")
                result.append(f"{ord_word}, {current_text.strip().rstrip('.')}.")
            current_num = m.group(1)
            current_text = m.group(2)
        else:
            current_text += " " + line

    if current_text:
        ord_word = ORDINALS.get(current_num, f"Poin {current_num}")
        result.append(f"{ord_word}, {current_text.strip().rstrip('.')}.")

    return " ".join(result) if result else text


def reconstruct_paragraph(text: str) -> str:
    """
    Rekonstruksi satu paragraf/blok menjadi teks naratif.

    Urutan:
    1. Jika heading → jadikan kalimat pengantar.
    2. Jika dominan bullet → reconstruct_bullets.
    3. Jika dominan numbered → reconstruct_numbered.
    4. Gabungkan baris yang terpotong secara heuristik.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""

    if looks_like_heading(text):
        return heading_to_sentence(text)

    bullet_count = sum(1 for l in lines if BULLET_RE.match(l))
    num_count = sum(1 for l in lines if NUMBER_RE.match(l))

    if bullet_count >= max(1, len(lines) * 0.35):
        return reconstruct_bullets(text)
    if num_count >= max(1, len(lines) * 0.35):
        return reconstruct_numbered(text)

    # Gabungkan baris yang terpotong di tengah kalimat
    merged: List[str] = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue

        prev = merged[-1]
        first = line.split()[0].lower() if line.split() else ""
        if prev[-1:] not in ".!?:" and (line[:1].islower() or first in CONJUNCTIONS):
            merged[-1] = prev + " " + line
        else:
            merged.append(line)

    return " ".join(merged)


def restore_punctuation(text: str) -> str:
    """Rapikan tanda baca dan pastikan paragraf panjang diakhiri titik."""
    paras: List[str] = []
    for p in PARA_SPLIT_RE.split(text):
        p = p.strip()
        if not p:
            continue
        p = PUNCT_AFTER_RE.sub(r"\1 ", p)
        p = PUNCT_BEFORE_RE.sub(r"\1", p)
        p = MULTI_SPACE_RE.sub(" ", p)
        if len(p.split()) > 5 and p[-1] not in ".!?":
            p += "."
        paras.append(p)
    return "\n\n".join(paras).strip()


# ═════════════════════════════════════════════════════════════════════════════
# 7. HEADING-AWARE + SEMANTIC CHUNKING
# ═════════════════════════════════════════════════════════════════════════════

_EMBEDDER = None


def sentence_tokenize(text: str) -> List[str]:
    """Sentence tokenizer Bahasa Indonesia dengan fallback regex."""
    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s.strip()]


def estimate_tokens(text: str) -> int:
    """Estimasi kasar jumlah token (BPE ≈ 1.35× jumlah kata)."""
    return int(len(text.split()) * 1.35)


def split_by_heading(paras: Sequence[str]) -> List[Dict[str, str]]:
    """
    Heading-aware segmentation.
    Heading pendek menjadi nama section, isi di bawahnya dikelompokkan
    sehingga chunk memiliki konteks section untuk pertanyaan yang lebih fokus.
    """
    sections: List[Dict[str, str]] = []
    current_title = ""
    current_parts: List[str] = []

    for p in paras:
        p = p.strip()
        if not p:
            continue

        is_heading_sentence = (
            p.lower().startswith("bagian ini membahas tentang") and len(p.split()) <= 12
        )
        is_heading_raw = looks_like_heading(p)

        if is_heading_raw or is_heading_sentence:
            if current_parts:
                sections.append({
                    "section": current_title,
                    "text": " ".join(current_parts).strip()
                })
                current_parts = []

            if is_heading_sentence:
                title = p.replace("Bagian ini membahas tentang", "").strip(" .")
            else:
                title = STRIP_NUM_RE.sub("", p).strip()
            current_title = title
            continue

        current_parts.append(p)

    if current_parts:
        sections.append({
            "section": current_title,
            "text": " ".join(current_parts).strip()
        })

    if not sections:
        joined = " ".join(p for p in paras if p.strip())
        sections = [{"section": "", "text": joined}]

    return sections


def get_embedder():
    """
    Lazy-load sentence-transformers model untuk semantic chunking.
    Jika tidak tersedia, return False agar chunking fallback ke sentence-aware.
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    except Exception:
        _EMBEDDER = False

    return _EMBEDDER


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity sederhana tanpa numpy."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_sentence_groups(
    sents: List[str], similarity_threshold: float = 0.48
) -> List[List[str]]:
    """
    Semantic chunking berbasis embedding kalimat.
    Jika similarity antar kalimat turun di bawah threshold, mulai group baru.
    Fallback ke satu group besar jika sentence-transformers tidak tersedia.
    """
    if len(sents) <= 1:
        return [sents]

    embedder = get_embedder()
    if not embedder:
        return [sents]

    try:
        vectors = embedder.encode(sents, normalize_embeddings=True).tolist()
    except Exception:
        return [sents]

    groups: List[List[str]] = [[sents[0]]]
    for i in range(1, len(sents)):
        sim = cosine_similarity(vectors[i - 1], vectors[i])
        if sim < similarity_threshold and len(groups[-1]) >= 2:
            groups.append([sents[i]])
        else:
            groups[-1].append(sents[i])
    return groups


def split_long_group(
    group: List[str], max_words: int = 110, overlap_sents: int = 1
) -> List[List[str]]:
    """
    Pecah group terlalu panjang agar chunk tidak terlalu luas.
    overlap_sents: berapa kalimat yang overlap antara chunk untuk menjaga konteks.
    """
    chunks: List[List[str]] = []
    current: List[str] = []
    count = 0

    for sent in group:
        sw = len(sent.split())
        if current and count + sw > max_words:
            chunks.append(current)
            current = current[-overlap_sents:] if overlap_sents > 0 else []
            count = sum(len(s.split()) for s in current)

        current.append(sent)
        count += sw

    if current:
        chunks.append(current)

    return chunks


def chunk_text(
    text: str,
    section: str = "",
    max_words: int = 120,
    min_words: int = 18,
) -> List[Dict[str, Any]]:
    """
    Chunking per section dengan 3 metode.
    Digabungkan jika terlalu pendek.
    """
    sents = sentence_tokenize(text)
    if not sents:
        return []

    groups = semantic_sentence_groups(sents)
    chunks: List[Dict[str, Any]] = []

    # Kumpulkan raw body
    raw_bodies = []
    for group in groups:
        for sub in split_long_group(group, max_words=max_words):
            body = " ".join(sub).strip()
            if body:
                raw_bodies.append(body)

    # Gabungkan jika terlalu pendek (< 25 kata)
    merged_bodies = []
    temp_body = []
    temp_count = 0
    for body in raw_bodies:
        wc = len(body.split())
        temp_body.append(body)
        temp_count += wc
        if temp_count >= 25:
            merged_bodies.append(" ".join(temp_body))
            temp_body = []
            temp_count = 0

    if temp_body:
        if merged_bodies:
            merged_bodies[-1] += " " + " ".join(temp_body)
        else:
            merged_bodies.append(" ".join(temp_body))

    for body in merged_bodies:
        if len(body.split()) < min_words:
            continue

        final_text = body
        if section and section.lower() not in final_text[:120].lower():
            final_text = f"{section}. {body}"

        chunks.append({
            "text": final_text,
            "section": section,
            "word_count": len(final_text.split()),
            "n_tokens": estimate_tokens(final_text),
            "n_sents": len(sentence_tokenize(final_text)),
            "method": "heading_semantic",
        })

    # Fallback
    if not chunks and len(" ".join(sents).split()) >= min_words:
        body = " ".join(sents)
        chunks.append({
            "text": f"{section}. {body}" if section else body,
            "section": section,
            "word_count": len(body.split()),
            "n_tokens": estimate_tokens(body),
            "n_sents": len(sents),
            "method": "heading_sentence_fallback",
        })

    return chunks


def score_content_chunk(text: str) -> float:
    """
    Skor kualitas chunk agar app.py memproses chunk paling layak lebih dulu.
    """
    low = text.lower()
    words = text.split()
    score = 0.0

    if 50 <= len(words) <= 120:
        score += 2.0
    elif 25 <= len(words) < 50:
        score += 1.0
    elif len(words) > 150:
        score -= 2.0
    else:
        score -= 1.0

    patterns = [
        " adalah ", " merupakan ", " yaitu ", " ialah ",
        " digunakan untuk ", " bertujuan untuk ", " berfungsi untuk ", 
        " fungsi ", " manfaat ", " tujuan ",
        " klik ", " pilih ", " tekan ", " buka ", " simpan ", " jalankan "
    ]
    score += sum(1.0 for p in patterns if p in low)

    if any(kw in low for kw in ["latihan", "soal", "daftar isi", "kata pengantar", "tugas", "evaluasi"]):
        score -= 4.0

    if _RUNNING_HEADER_HINT_RE.search(low):
        score -= 3.0

    table_hits = len(re.findall(r"\b(sub\s*menu|tombol|fungsinya|keterangan\s*[a-z]?$|sebagai\s+berikut\s*:?|toolbar|menu\s+bar|page\s+setup|print\s+preview|font\s+styles)\b", low))
    if table_hits >= 2:
        score -= 3.0
    if table_hits >= 4:
        score -= 5.0
        
    shortcut_hits = len(re.findall(r"\b(ctrl|alt|shift)\s*\+", low))
    if shortcut_hits >= 2:
        score -= 4.0
    if shortcut_hits >= 4:
        score -= 6.0

    return score


# ═════════════════════════════════════════════════════════════════════════════
# 8. PICK ANSWER DAN QUESTION TYPE
# ═════════════════════════════════════════════════════════════════════════════

DEFINITION_PATTERNS = [
    re.compile(
        r"\b(?P<term>[A-ZÀ-ÿ0-9][A-Za-zÀ-ÿ0-9\s/\-]{2,60}?)\s+"
        r"(?:adalah|merupakan|ialah|yaitu)\s+(?P<ans>[^.?!]{8,220})",
        re.IGNORECASE,
    ),
]
FUNCTION_PATTERNS = [
    re.compile(
        r"\b(?P<term>[A-ZÀ-ÿ0-9][A-Za-zÀ-ÿ0-9\s/\-]{2,60}?)\s+"
        r"(?:digunakan untuk|berfungsi untuk|bertujuan untuk|berguna untuk)\s+"
        r"(?P<ans>[^.?!]{8,220})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btujuan\s+(?P<term>[A-Za-zÀ-ÿ0-9\s/\-]{2,60}?)\s+"
        r"(?:adalah|yaitu)\s+(?P<ans>[^.?!]{8,220})",
        re.IGNORECASE,
    ),
]
COMPONENT_PATTERNS = [
    re.compile(
        r"\b(?P<term>[A-ZÀ-ÿ0-9][A-Za-zÀ-ÿ0-9\s/\-]{2,60}?)\s+"
        r"(?:terdiri dari|meliputi|mencakup)\s+(?P<ans>[^.?!]{8,240})",
        re.IGNORECASE,
    ),
]

DATE_RE = re.compile(
    r"\b(\d{1,2}\s+(januari|februari|maret|april|mei|juni|juli|agustus|"
    r"september|oktober|november|desember)\s+\d{4}|\d{4})\b",
    re.IGNORECASE,
)
NUMBER_RE_ANSWER = re.compile(
    r"\b\d+(?:[.,]\d+)*(?:\s*(?:%|persen|km2|km|m|cm|kg|orang|unit|tahun|ha))?\b",
    re.IGNORECASE,
)
LOCATION_CUE_RE = re.compile(
    r"\b(di|ke|dari)\s+([A-Z][A-Za-zÀ-ÿ]+(?:\s+[A-Z][A-Za-zÀ-ÿ]+){0,3})\b"
)
PERSON_CUE_RE = re.compile(
    r"\b(?:oleh|bernama|nama|tokoh)\s+([A-Z][A-Za-zÀ-ÿ]+(?:\s+[A-Z][A-Za-zÀ-ÿ]+){1,3})\b"
)


def clean_option_text(text: str) -> str:
    """
    Bersihkan teks opsi jawaban / answer candidate dari noise.

    Menghapus:
    - Running header/footer yang menempel (Bahan Ajar Aplikasi Komputer, dll)
    - Frasa pengantar tidak informatif (keterangan, sebagai berikut, dll)
    - Duplikasi kata berurutan (Transisi Transisi → Transisi)
    - Teks terlalu panjang (>32 kata dipotong)
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[\u00ad\ufffd]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \n\t:;,.•-–—")

    # Hapus sisa running header/footer
    text = re.sub(
        r"^\d{1,4}\s+bahan\s+ajar\s+aplikasi\s+komputer\s*", "",
        text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*bahan\s+ajar\s+aplikasi\s+komputer\s*$", "",
        text, flags=re.IGNORECASE,
    )
    text = re.sub(r"^frizka\s+fitriana.*?\d{0,4}\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*frizka\s+fitriana.*$", "", text, flags=re.IGNORECASE)

    # Hapus frasa pengantar dari tabel/parsing buruk
    text = re.sub(
        r"^(dengan\s+penjelasan|penjelasan|keterangan|sebagai\s+berikut)\s*:?\s*",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" \n\t:;,.•-–—")

    # Deduplicate kata berurutan: "Transisi Transisi" → "Transisi"
    words = text.split()
    dedup = []
    for w in words:
        if dedup and dedup[-1].lower().strip(".,") == w.lower().strip(".,"):
            continue
        dedup.append(w)
    text = " ".join(dedup)

    if len(text.split()) > 32:
        text = " ".join(text.split()[:32])
    return text.strip()


def is_clean_distractor(text: str) -> bool:
    """
    Cek apakah text layak menjadi opsi jawaban/distractor.
    """
    text = clean_option_text(text)
    if not text:
        return False
    low = text.lower()
    words = text.split()

    if len(text) < 3 and not any(ch.isdigit() for ch in text) and text.upper() != text:
        return False
    if len(words) > 35:
        return False

    bad_substrings = [
        "[error", "tidak dapat", "semua jawaban", "tidak ada jawaban",
        "bahan ajar aplikasi komputer", "frizka fitriana", "daftar isi",
        "kata pengantar", "hak cipta", "diterbitkan", "isbn",
        "sebagai berikut", "keterangan", "sub menu", "tombol", "toolbar standart",
        "pilihan ini tidak tersedia", "belum dapat ditentukan", "tidak ada informasi",
        "fungsinya"
    ]
    if any(b in low for b in bad_substrings):
        return False

    if len(re.findall(r"\b(sub\s*menu|tombol|fungsinya|toolbar|menu\s+bar)\b", low)) >= 2:
        return False

    if len(re.findall(r"\b(ctrl|alt|shift)\s*\+", low)) >= 1 and len(words) < 12:
        return False

    if "," in low and len(words) > 28:
        return False
        
    generic = {"data", "sistem", "proses", "informasi", "metode", "konsep", "materi", "hal", "tools"}
    if low in generic:
        return False
        
    bad_prefixes = ["yang ", "dimana ", "sebagai ", "dengan ", "dan ", "atau "]
    if any(low.startswith(p) for p in bad_prefixes):
        return False

    return True


def infer_question_type(answer: str, context: str = "", target_term: str = "") -> str:
    """
    Infer question_type untuk format input model IndoT5-QTYPE.
    """
    ans = clean_option_text(answer)
    low_ans = ans.lower()
    low_ctx = context.lower()

    if DATE_RE.search(ans):
        return "kapan"

    if NUMBER_RE_ANSWER.fullmatch(ans) or (re.search(r"\d", ans) and len(ans.split()) <= 6):
        return "berapa"

    loc_match = LOCATION_CUE_RE.search(context)
    if loc_match and ans in loc_match.group(0):
        return "di_mana"

    per_match = PERSON_CUE_RE.search(context)
    if per_match and ans in per_match.group(0):
        return "siapa"

    if any(x in low_ctx for x in ["langkah", "cara ", "prosedur", "klik", "pilih", "tekan", "buka", "simpan", "masukkan", "jalankan", "atur", "gunakan"]):
        if any(v in ans.lower() for v in ["klik", "pilih", "tekan", "buka", "simpan", "masukkan", "jalankan", "atur", "gunakan", "dengan"]):
            return "bagaimana"

    if any(x in low_ctx for x in ["digunakan untuk", "berfungsi untuk", "fungsi ", "tujuan ", "manfaat "]):
        return "tugas_fungsi"

    if target_term and target_term.lower() in low_ctx:
        if any(x in low_ctx for x in ["adalah", "merupakan", "yaitu", "ialah"]):
            return "definisi"
            
    if any(x in low_ctx for x in ["adalah", "merupakan", "yaitu", "ialah"]):
        return "definisi"

    return "apa" 


def answer_candidate_score(
    answer: str, source: str, context: str, question_type: str
) -> float:
    """
    Scoring kandidat answer.
    """
    ans = clean_option_text(answer)
    words = ans.split()
    low = ans.lower()
    ctx_low = context.lower()

    score = 0.0
    if source == "definition":
        score += 4.0
    elif source == "function":
        score += 4.0
    elif source == "procedure":
        score += 4.0
    elif source == "component":
        score += 3.0
    elif source in {"date", "number", "person", "location"}:
        score += 2.5
    elif source.startswith("fallback"):
        score += 1.0

    if 5 <= len(words) <= 22:
        score += 2.0
    elif 2 <= len(words) < 5:
        score += 0.6
    elif len(words) > 30:
        score -= 3.0
    elif len(words) <= 1 and question_type not in {"kapan", "berapa"}:
        score -= 2.5

    if any(ch.isdigit() for ch in ans) and question_type in {"berapa", "kapan"}:
        score += 1.0

    generic = {"data", "sistem", "proses", "informasi", "metode", "konsep", "materi", "hal", "tools"}
    if low in generic:
        score -= 4.0

    bad = [
        "bahan ajar aplikasi komputer", "frizka fitriana", "sebagai berikut",
        "keterangan", "sub menu", "tom bol", "fungsinya", "toolbar", "menu bar",
    ]
    if any(b in low for b in bad):
        score -= 5.0

    table_hits = len(re.findall(r"\b(sub\s*menu|tombol|fungsinya|keterangan\s*[a-z]?$|sebagai\s+berikut\s*:?|toolbar|menu\s+bar|page\s+setup|print\s+preview|font\s+styles)\b", low))
    if table_hits >= 2:
        score -= 3.0
    if table_hits >= 4:
        score -= 5.0

    shortcut_hits = len(re.findall(r"\b(ctrl|alt|shift)\s*\+", low))
    if shortcut_hits >= 1:
        score -= 3.0

    if "," in ans and len(words) > 26:
        score -= 1.5

    return score


def clean_target_term(term: str) -> str:
    """
    Bersihkan target term dari hasil regex yang terlalu panjang/duplikatif.
    """
    term = clean_option_text(term)
    if not term:
        return term

    # Hapus kata pengantar heading generik
    term = re.sub(
        r"^(bagian ini membahas tentang|pengertian|pengenalan)\s+",
        "", term, flags=re.IGNORECASE,
    ).strip()

    # Buang awalan tidak jelas
    term = re.sub(r"^(dan|atau|serta|tentang|yang|itu|ini|terakhir ini|kolom dapat)\s+", "", term, flags=re.IGNORECASE).strip()

    # Deduplicate phrase berurutan sepanjang 1-3 kata
    words = term.split()
    for n in (3, 2, 1):
        if (len(words) >= 2 * n
                and [w.lower() for w in words[-2*n:-n]] == [w.lower() for w in words[-n:]]):
            words = words[:-n]
            term = " ".join(words)
            break

    if len(words) > 8:
        term = " ".join(words[-7:])
        term = re.sub(r"^(dan|atau|serta|tentang|yang|itu|ini)\s+", "", term, flags=re.IGNORECASE)

    if " dan " in term.lower() and len(term.split()) > 4:
        parts = re.split(r"\s+dan\s+", term, flags=re.IGNORECASE)
        term = parts[-1].strip()

    term = term.strip()
    bad_terms = {"itu", "ini", "terakhir ini", "kolom dapat", "0 windows", "baris kerja"}
    if term.lower() in bad_terms:
        return ""

    return clean_option_text(term)


def extract_pattern_candidates(context: str) -> List[Dict[str, Any]]:
    """Ekstrak kandidat answer dari pola definisi/fungsi/komponen/prosedur."""
    candidates: List[Dict[str, Any]] = []

    for pat in DEFINITION_PATTERNS:
        for m in pat.finditer(context):
            term = clean_target_term(m.group("term"))
            ans = clean_option_text(m.group("ans"))
            if term:
                candidates.append({
                    "answer": ans, "target_term": term,
                    "question_type": "definisi", "source": "definition",
                })

    for pat in FUNCTION_PATTERNS:
        for m in pat.finditer(context):
            term = clean_target_term(m.group("term"))
            ans = clean_option_text(m.group("ans"))
            if term:
                candidates.append({
                    "answer": ans, "target_term": term,
                    "question_type": "tugas_fungsi", "source": "function",
                })

    for pat in COMPONENT_PATTERNS:
        for m in pat.finditer(context):
            term = clean_target_term(m.group("term"))
            ans = clean_option_text(m.group("ans"))
            if term:
                candidates.append({
                    "answer": ans, "target_term": term,
                    "question_type": "apa", "source": "component",
                })
                
    proc_pattern = re.compile(r"\b(?:cara|untuk|langkah)\s+(?P<term>[A-Za-zÀ-ÿ0-9\s/\-]{2,60}?)[,:]?\s+(?P<ans>(?:klik|pilih|tekan|buka|simpan|masukkan|jalankan|atur|gunakan)\s+[^.?!]{5,150})", re.IGNORECASE)
    for m in proc_pattern.finditer(context):
        term = clean_target_term(m.group("term"))
        ans = clean_option_text(m.group("ans"))
        if term:
            candidates.append({
                "answer": ans, "target_term": term,
                "question_type": "bagaimana", "source": "procedure",
            })

    return candidates


def extract_entity_candidates(context: str) -> List[Dict[str, Any]]:
    """Kandidat dari angka/tanggal/lokasi/orang yang tersebut di context."""
    candidates: List[Dict[str, Any]] = []

    for m in DATE_RE.finditer(context):
        ans = clean_option_text(m.group(0))
        candidates.append({
            "answer": ans, "target_term": "",
            "question_type": "kapan", "source": "date",
        })

    for m in NUMBER_RE_ANSWER.finditer(context):
        ans = clean_option_text(m.group(0))
        if len(ans) >= 2:
            candidates.append({
                "answer": ans, "target_term": "",
                "question_type": "berapa", "source": "number",
            })

    for m in LOCATION_CUE_RE.finditer(context):
        ans = clean_option_text(m.group(2))
        candidates.append({
            "answer": ans, "target_term": "",
            "question_type": "di_mana", "source": "location",
        })

    for m in PERSON_CUE_RE.finditer(context):
        ans = clean_option_text(m.group(1))
        candidates.append({
            "answer": ans, "target_term": "",
            "question_type": "siapa", "source": "person",
        })

    return candidates


def get_focused_context(context: str, answer: str, window_sents: int = 2, section: str = "") -> str:
    """
    Ambil kalimat di sekitar answer dari context chunk.
    """
    sec_match = re.match(r"^([^.]{2,60})\.\s+", context)
    if sec_match and not section:
        section = sec_match.group(1).strip()
        
    sents = sentence_tokenize(context)
    if not sents:
        return context

    ans_low = clean_option_text(answer).lower()
    ans_tokens = ans_low.split()
    idx = 0

    for i, sent in enumerate(sents):
        sent_low = sent.lower()
        if ans_low and ans_low in sent_low:
            idx = i
            break
        if len(ans_tokens) >= 4:
            hit = sum(1 for t in ans_tokens if t in sent_low)
            if hit >= max(2, int(len(ans_tokens) * 0.45)):
                idx = i
                break

    start = max(0, idx - window_sents)
    end   = min(len(sents), idx + window_sents + 1)
    focused = " ".join(sents[start:end]).strip()
    
    focused_words = focused.split()
    if len(focused_words) > 150:
        focused = " ".join(focused_words[:150]) + "..."
        
    if section and section.lower() not in focused[:120].lower():
        focused = f"[Section: {section}] {focused}"
        
    return focused

focused_context_around_answer = get_focused_context


def pick_answer(context: str) -> Optional[Dict[str, Any]]:
    """
    Pilih answer terbaik dari chunk context.

    Return dict (bukan string) berisi:
    {
        "answer": str,          — jawaban benar dari dokumen
        "question_type": str,   — label untuk model IndoT5-QTYPE
        "focused_context": str, — kalimat sekitar answer untuk model
        "target_term": str,     — subjek pertanyaan
        "score": float,         — skor kandidat
        "source": str,          — asal kandidat (definition/function/dll)
        "reason": str,          — alasan mengapa diterima/ditolak
    }
    """
    if not context or len(context.split()) < 8:
        return None

    # Ektrak section jika ada
    section = ""
    sec_match = re.match(r"^([^.]{2,60})\.\s+", context)
    if sec_match:
        section = sec_match.group(1).strip()

    candidates = extract_pattern_candidates(context) + extract_entity_candidates(context)

    # Fallback: ambil setelah "adalah/merupakan/yaitu/ialah" dari kalimat awal
    if not candidates:
        sents = sentence_tokenize(context)
        for sent in sents[:3]:
            m = re.search(
                r"(?:adalah|merupakan|yaitu|ialah)\s+([^.!?]{8,180})",
                sent, flags=re.IGNORECASE,
            )
            if m:
                ans = clean_option_text(m.group(1))
                candidates.append({
                    "answer": ans, "target_term": "",
                    "question_type": "definisi", "source": "fallback_definition",
                })

    scored: List[Dict[str, Any]] = []
    seen: set[str] = set()
    reasons: List[str] = []

    for cand in candidates:
        ans = clean_option_text(cand.get("answer", ""))
        term = cand.get("target_term", "")

        if not ans:
            continue

        if not is_clean_distractor(ans):
            reasons.append(f"Rejected: Not clean distractor ({ans[:30]})")
            continue

        if term and term.lower() in ans.lower() and len(ans.split()) < 6:
            reasons.append(f"Rejected: Answer too similar to term ({ans[:30]})")
            continue

        if ans.lower() in seen:
            continue
        seen.add(ans.lower())

        qtype = cand.get("question_type") or infer_question_type(
            ans, context, term
        )
        score = answer_candidate_score(ans, cand.get("source", ""), context, qtype)
        focused = get_focused_context(context, ans, window_sents=2, section=section)

        item = dict(cand)
        item.update({
            "answer": ans,
            "question_type": qtype,
            "focused_context": focused,
            "score": score,
            "reason": "OK" if score >= 0 else "Low score"
        })
        scored.append(item)

    if not scored:
        if reasons:
            return {"answer": "", "question_type": "", "focused_context": "", "target_term": "", "score": -1.0, "reason": " | ".join(reasons[:3])}
        return None

    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]
    if best["score"] < 0:
        best["reason"] = "Score < 0"
    return best


# ═════════════════════════════════════════════════════════════════════════════
# 9. VALIDATOR DAN DISTRACTOR HEURISTIC
# ═════════════════════════════════════════════════════════════════════════════

def detect_question_word(question: str) -> str:
    """Deteksi kata tanya di awal pertanyaan."""
    q = question.lower().strip()
    if q.startswith(("siapa", "siapakah")):
        return "siapa"
    if q.startswith(("kapan", "kapankah")):
        return "kapan"
    if q.startswith(("berapa", "berapakah")):
        return "berapa"
    if q.startswith(("di mana", "dimana", "ke mana", "kemana", "dari mana")):
        return "di_mana"
    if q.startswith(("mengapa", "kenapa")):
        return "mengapa"
    if q.startswith(("bagaimana", "bagaimanakah")):
        return "bagaimana"
    if "tugas" in q or "fungsi" in q or "tujuan" in q:
        return "tugas_fungsi"
    if "dimaksud" in q or "definisi" in q or "pengertian" in q:
        return "definisi"
    if q.startswith(("apa", "apakah")):
        return "apa"
    return "unknown"


def heuristic_validate_question(
    question: str, answer: str, context: str, question_type: str
) -> Dict[str, Any]:
    """
    Validator heuristic jika Qwen tidak aktif.

    Mengecek:
    1. Pertanyaan tidak terlalu pendek.
    2. Answer muncul di context (fuzzy).
    3. Kata tanya sesuai dengan question_type.

    Return dict kompatibel dengan format output validator Qwen.
    """
    q = (question or "").strip()
    a = clean_option_text(answer)
    c = (context or "").lower()
    score = 3
    reasons: List[str] = []

    if not q or len(q.split()) < 3:
        score -= 2
        reasons.append("pertanyaan terlalu pendek")

    if a.lower() not in c:
        atoks = [t for t in a.lower().split() if len(t) > 3]
        hit = sum(1 for t in atoks if t in c)
        if atoks and hit / len(atoks) < 0.45:
            score -= 1
            reasons.append("answer kurang terlihat di context")

    qword = detect_question_word(q)
    expected = question_type

    ok = False
    if expected == qword:
        ok = True
    elif expected == "definisi" and qword in {"apa", "definisi"}:
        ok = True
    elif expected == "tugas_fungsi" and qword in {"apa", "tugas_fungsi", "mengapa"}:
        ok = True
    elif expected == "apa" and qword in {"apa", "definisi", "tugas_fungsi"}:
        ok = True

    if not ok:
        score -= 1
        reasons.append(f"kata tanya '{qword}' kurang cocok dengan tipe '{expected}'")

    if qword in {"apa", "definisi"} and question_type in {"kapan", "berapa", "siapa", "di_mana"}:
        score -= 2
        reasons.append("pertanyaan definisi tidak cocok untuk answer faktual")

    return {
        "valid": score >= 2,
        "question": q,
        "reason": "; ".join(reasons) if reasons else "lolos validasi heuristic",
        "source": "heuristic",
    }


def generate_heuristic_distractors(
    answer: str, context: str, question_type: str = "apa", n: int = 3
) -> List[str]:
    """
    Distractor fallback tanpa LLM.

    Strategi:
    - Untuk pertanyaan faktual (berapa/kapan/di_mana/siapa): cari entitas sejenis
      dari context.
    - Untuk pertanyaan lain: cari frasa setelah pola definisi/fungsi lain,
      atau kalimat pendek dari context yang berbeda dari jawaban benar.
    - Tidak pakai "semua jawaban benar" kecuali benar-benar fallback terakhir.
    """
    answer = clean_option_text(answer)
    candidates: List[str] = []

    if question_type == "berapa":
        candidates.extend(
            clean_option_text(m.group(0)) for m in NUMBER_RE_ANSWER.finditer(context)
        )
    elif question_type == "kapan":
        candidates.extend(
            clean_option_text(m.group(0)) for m in DATE_RE.finditer(context)
        )
    elif question_type == "di_mana":
        candidates.extend(
            clean_option_text(m.group(2)) for m in LOCATION_CUE_RE.finditer(context)
        )
    elif question_type == "siapa":
        candidates.extend(
            clean_option_text(m.group(1)) for m in PERSON_CUE_RE.finditer(context)
        )
    else:
        # Ambil frasa dari pola definisi/fungsi lain di context
        for cand in extract_pattern_candidates(context):
            candidates.append(clean_option_text(cand.get("answer", "")))

        # Fallback: kalimat pendek dari context
        for sent in sentence_tokenize(context):
            s = clean_option_text(sent)
            if 4 <= len(s.split()) <= 20:
                candidates.append(s)

    clean: List[str] = []
    for c in candidates:
        c = clean_option_text(c)
        if not is_clean_distractor(c):
            continue
        if c.lower() == answer.lower():
            continue
        if c.lower() in {x.lower() for x in clean}:
            continue
        clean.append(c)
        if len(clean) >= n:
            break

    return clean[:n]


# Alias untuk kompatibilitas backward
validate_qa_heuristic = heuristic_validate_question
generate_misleading_distractors = generate_heuristic_distractors


# ═════════════════════════════════════════════════════════════════════════════
# 10. MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def merge_short_paragraphs(paras: List[str], min_words: int = 28) -> List[str]:
    """Gabungkan paragraf terlalu pendek agar konteks tidak terlalu pecah."""
    if not paras:
        return []
    merged: List[str] = []
    for p in paras:
        if merged and len(merged[-1].split()) < min_words:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def full_preprocess_pipeline(raw_text: str) -> Dict[str, Any]:
    """
    Pipeline preprocessing lengkap untuk teks dari dokumen kuliah.

    Tahapan:
    01_cleaned          — cleaning unicode, control char, spasi, dehyphenation
    02_noise_filtered   — filter paragraf non-edukatif (TOC, metadata, dll)
    03_task_block_removed — hapus blok tugas/latihan/pertanyaan
    04_reconstructed    — heading/bullet/list → kalimat naratif
    05_normalized       — singkatan informal → bentuk lengkap
    06_abbreviation_expanded — akronim kapital → definisi (opsional Wikipedia)
    07_punctuation_restored — rapikan tanda baca dan titik akhir paragraf
    08_chunks           — heading-aware + semantic chunking

    Return dict berisi semua stage intermediate untuk debug,
    ditambah 'chunks' yang siap dikirim ke pick_answer + model.
    """
    raw_text = raw_text or ""

    # Tahap 1: cleaning awal
    cleaned = clean_text(raw_text)

    # Tahap 2: split paragraf
    raw_paras = [p.strip() for p in PARA_SPLIT_RE.split(cleaned) if p.strip()]

    # Tahap 3a: hapus blok tugas/latihan/pertanyaan setelah heading terkait
    task_filtered = remove_task_blocks(raw_paras)

    # Tahap 3b: filter paragraf non-edukatif (running header, TOC, metadata)
    filtered = [p for p in task_filtered if is_educational_paragraph(p)]

    # Tahap 4: gabungkan paragraf pendek
    merged = merge_short_paragraphs(filtered, min_words=28)

    # Tahap 5: rekonstruksi heading, bullet, numbered list → narasi
    reconstructed = [reconstruct_paragraph(p) for p in merged]
    reconstructed = [p for p in reconstructed if p and len(p.split()) >= 5]
    recon_text = "\n\n".join(reconstructed)

    # Tahap 6: normalisasi singkatan informal
    normalized = normalize_informal_abbreviations(recon_text)

    # Tahap 7: ekspansi akronim (inline + Wikipedia optional)
    expanded = expand_abbreviations(normalized)

    # Tahap 8: restorasi tanda baca
    restored = restore_punctuation(expanded)

    # Tahap 9: heading-aware + semantic chunking
    final_paras = [p.strip() for p in PARA_SPLIT_RE.split(restored) if p.strip()]
    sections = split_by_heading(final_paras)

    chunks: List[Dict[str, Any]] = []
    cid = 0
    for sec in sections:
        section_chunks = chunk_text(sec["text"], section=sec.get("section", ""))
        for ch in section_chunks:
            ch["chunk_id"] = cid
            ch["content_score"] = score_content_chunk(ch["text"])
            cid += 1
            chunks.append(ch)

    # Fallback terakhir jika semua chunking gagal
    if not chunks and restored.strip():
        body = restored.strip()
        chunks = [{
            "chunk_id": 0,
            "text": body,
            "section": "",
            "word_count": len(body.split()),
            "n_tokens": estimate_tokens(body),
            "n_sents": len(sentence_tokenize(body)),
            "method": "full_text_fallback",
            "content_score": score_content_chunk(body),
        }]

    pipeline_stages = {
        "01_raw_words": len(raw_text.split()),
        "02_cleaned_words": len(cleaned.split()),
        "03_raw_paragraphs": len(raw_paras),
        "04_after_task_block_filter": len(task_filtered),
        "05_filtered_paragraphs": len(filtered),
        "06_merged_paragraphs": len(merged),
        "07_sections": len(sections),
        "08_chunks": len(chunks),
    }

    return {
        "raw": raw_text,
        "cleaned": cleaned,
        "noise_filtered": "\n\n".join(filtered),
        "task_block_removed": "\n\n".join(task_filtered),
        "reconstructed": recon_text,
        "normalized": normalized,
        "expanded": expanded,
        "restored": restored,
        "sections": sections,
        "chunks": chunks,
        "pipeline_stages": pipeline_stages,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Pipeline utama
    "full_preprocess_pipeline",
    # Parsing helpers
    "remove_repeated_headers_footers",
    "is_noise_unit",
    "remove_task_blocks",
    # Answer selection
    "pick_answer",
    "infer_question_type",
    "get_focused_context",
    "focused_context_around_answer",  # alias
    # Cleaning helpers
    "clean_option_text",
    "is_clean_distractor",
    # Validator & distractor heuristic
    "heuristic_validate_question",
    "generate_heuristic_distractors",
    # Aliases backward compat
    "validate_qa_heuristic",
    "generate_misleading_distractors",
    # Regex publik yang dipakai app.py
    "_TOC_LINE_RE",
    "_COLOPHON_RE",
    "_PUBLICATION_META_RE",
    "_RUNNING_HEADER_HINT_RE",
]
