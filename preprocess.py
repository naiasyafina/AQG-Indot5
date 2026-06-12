"""
preprocess.py — AQG Preprocessing Pipeline (Pure Logic, NO Streamlit)

Pipeline 9 tahap untuk mempersiapkan teks dari dokumen (PDF/DOCX/PPTX)
sebelum masuk ke model IdT5 untuk generate pertanyaan.

Novelty utama:
  1. expand_abbreviations()        — NLP-based + Wikipedia API (bukan hardcoded dict)
  2. normalize_informal_abbreviations() — normalisasi singkatan informal Bahasa Indonesia
     (yg→yang, dgn→dengan, krn→karena, tsb→tersebut, dll.)
  3. heading_to_sentence()         — konversi heading slide ke kalimat pengantar
  4. attach_section_context()      — prefix section ke setiap chunk
  5. filter_reconstructed_sentences() — buang fragmen pendek post-rekonstruksi
  6. semantic chunking             — potong berdasarkan perubahan makna (cosine sim.)
"""

import re
import unicodedata
import logging
from functools import lru_cache
from typing import List, Dict, Optional

import nltk
import requests

# ── NLTK Setup ───────────────────────────────────────────────────────────────
for _pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# PRE-COMPILED REGEX
# ════════════════════════════════════════════════════════════════════════════

_NOISE_RE = [
    re.compile(r"^\d+\s*$"),
    re.compile(r"^[ivxIVX]+\s*$"),
    re.compile(r"\f"),
    re.compile(r"^\d+\s+Bahan\s+Ajar", re.IGNORECASE),
    re.compile(r"Bahan\s+Ajar.*Komputer\s*$", re.IGNORECASE),
    re.compile(r"Frizka\s+Fitriana", re.IGNORECASE),
]

# Pola untuk mendeteksi halaman TOC (Daftar Isi)
_TOC_LINE_RE = re.compile(
    r"^[A-Za-z0-9\s\.\-–]+\s{2,}\d+\s*$"
    r"|^[A-Za-z0-9\s]+\.*\s+[ivxIVX]+\s*$"
    r"|^\s*[A-Z]\.\s+.{5,}\s+\d+\s*$"
)

# Header/footer struktural yang tidak mengandung konten
_STRUCTURAL_HEADER_RE = re.compile(
    r"^(kata\s+pengantar|daftar\s+isi|daftar\s+pustaka|daftar\s+gambar|"
    r"daftar\s+tabel|daftar\s+singkatan|prakata|sanwacana|persembahan|"
    r"halaman\s+judul|lembar\s+pengesahan|abstrak|abstract|"
    r"biodata\s+penulis|tentang\s+penulis|tentang\s+penyusun|"
    r"profil\s+penulis|riwayat\s+hidup)\s*$",
    re.IGNORECASE
)

# Pola kota + tahun (colophon): "Tembilahan, 2020"
_COLOPHON_RE = re.compile(
    r"^[A-Z][a-zA-Z\s]+,\s*(19|20)\d{2}\s*$"
)

# Pola ISBN, copyright, penerbit
_PUBLICATION_META_RE = re.compile(
    r"isbn|hak\s+cipta|copyright|©|diterbitkan\s+oleh|penerbit|"
    r"cetakan\s+(ke|pertama|kedua)|all\s+rights\s+reserved",
    re.IGNORECASE
)

# TOC inline — sudah digabung jadi satu baris panjang oleh clean_text
_TOC_INLINE_RE = re.compile(
    r"(?:kata\s+pengantar|daftar\s+isi|daftar\s+pustaka)"
    r".{0,300}"
    r"(?:i{1,4}v?|vi{0,3}|ix|\d{1,3})\s*(?:$|[A-Z])",
    re.IGNORECASE
)

BULLET_RE       = re.compile(r"^[•\-\*\–\→●○►]\s+")
BULLET_CAP_RE   = re.compile(r"^[•\-\*\–\→●○►]\s+(.*)")
NUMBER_RE       = re.compile(r"^(\d+\.\s+|[a-zA-Z]\.\s+)")
NUMBER_CAP_RE   = re.compile(r"^(\d+)\.\s+(.*)")
COMPLETE_RE     = re.compile(r"[.!?]\s*$")
NEXT_LINE_RE    = re.compile(r"^(\d+\.|[•\-\*])")
CTRL_CHAR_RE    = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE  = re.compile(r" {2,}")
DEHYPHEN_RE     = re.compile(r"(\w)-\n(\w)")
PUNCT_AFTER_RE  = re.compile(r"([.,;:!?])([^\s\"'])")
PUNCT_BEFORE_RE = re.compile(r"\s+([.,;:!?])")
CAPITALIZE_RE   = re.compile(r"(?<=[.!?])\s+([a-z])")
STRIP_NUM_RE    = re.compile(r"^(\d+\.|[a-zA-Z]\.)\s*")

_ABBR_EXCEPT  = r"(?<!dll)(?<!dsb)(?<!dkk)(?<!yth)(?<!sdr)(?<!no)(?<!vol)(?<!hal)(?<!hlm)(?<!Dr)(?<!Mr)(?<!Mrs)(?<!Prof)(?<!Jr)(?<!Sr)"
SENT_SPLIT    = re.compile(_ABBR_EXCEPT + r"(?<=[.!?])\s+")
PARA_SPLIT_RE = re.compile(r"\n{2,}")

# Pola singkatan kapital 2–6 karakter: CPU, I/O, GPU, NLP, BIOS, AI
ABBR_DETECT_RE = re.compile(r"\b([A-Z][A-Z/]{1,5})\b")

_TASK_KEYWORDS = frozenset({"tugas", "latihan", "soal", "buatlah",
                            "presentasikan", "kerjakan", "diskusikan"})

ORDINALS = {
    "1": "Pertama", "2": "Kedua",  "3": "Ketiga",   "4": "Keempat",
    "5": "Kelima",  "6": "Keenam", "7": "Ketujuh",  "8": "Kedelapan",
    "9": "Kesembilan",
}
CONJUNCTIONS = frozenset({"dan", "atau", "serta", "yang", "untuk", "dengan",
                           "pada", "ke", "dari", "oleh", "dalam", "ini",
                           "itu", "juga", "namun"})


# ════════════════════════════════════════════════════════════════════════════
# NOVELTY BARU — Normalisasi Singkatan Informal Bahasa Indonesia
# ════════════════════════════════════════════════════════════════════════════
#
# MENGAPA INI PERLU:
#   Singkatan informal seperti "yg", "dgn", "tsb" sangat umum di dokumen
#   akademik Indonesia (slide kuliah, diktat, hasil OCR). Singkatan ini
#   TIDAK dikenali oleh ABBR_DETECT_RE (huruf kecil semua) sehingga lolos
#   ke model T5 tanpa di-expand — padahal model IDK-MRC dilatih dengan
#   teks formal.
#
#   Pendekatan ini berbeda dari expand_abbreviations() (yang handle akronim
#   kapital seperti CPU, AI, BPJS). Ini handle kata-kata pendek informal.
#
#   Diproses SEBELUM expand_abbreviations() agar teks sudah bersih saat
#   Wikipedia lookup dijalankan.

_INFORMAL_ABBR: Dict[str, str] = {
    # Konjungsi / preposisi
    "yg"  : "yang",
    "dgn" : "dengan",
    "dg"  : "dengan",
    "krn" : "karena",
    "krna": "karena",
    "utk" : "untuk",
    "dlm" : "dalam",
    "thd" : "terhadap",
    "thdp": "terhadap",
    "pd"  : "pada",
    "dr"  : "dari",
    "sbg" : "sebagai",
    "spt" : "seperti",
    "spy" : "supaya",
    "stlh": "setelah",
    "sblm": "sebelum",
    "sdh" : "sudah",
    "blm" : "belum",
    "jg"  : "juga",
    "lg"  : "lagi",
    "lbh" : "lebih",
    "krg" : "kurang",
    "kpd" : "kepada",
    "tsb" : "tersebut",
    "dpt" : "dapat",
    "bs"  : "bisa",
    "ttg" : "tentang",
    "mjd" : "menjadi",
    "adl" : "adalah",
    "tdk" : "tidak",
    "hrs" : "harus",
    "sm"  : "sama",
    "byk" : "banyak",
    "jml" : "jumlah",
    "jwb" : "jawab",
    "prlu": "perlu",
    "msk" : "masuk",
    "klau": "kalau",
    "klo" : "kalau",
    "kl"  : "kalau",
    "sy"  : "saya",
    "mrk" : "mereka",
    "mk"  : "maka",
    "tnp" : "tanpa",
    # Kata kerja / sifat umum
    "dpt" : "dapat",
    "mll" : "melalui",
    "ttg" : "tentang",
    "brs" : "bersama",
    "scr" : "secara",
    "trs" : "terus",
    "sll" : "selalu",
    "plg" : "paling",
    # Singkatan akademik informal
    "dkk" : "dan kawan-kawan",
    "dst" : "dan seterusnya",
    "dsb" : "dan sebagainya",
    # "dll" → "dan lain-lain" (sudah ada di _ABBR_EXCEPT, tetap di-expand agar teks natural)
    "dll" : "dan lain-lain",
}

# Compile sekali agar tidak re-compile tiap kali fungsi dipanggil.
# Word boundary (\b) wajib agar "yang" tidak tertrigger di dalam kata lain.
_INFORMAL_ABBR_RE: Dict[str, re.Pattern] = {
    abbr: re.compile(r"\b" + re.escape(abbr) + r"\b", re.IGNORECASE)
    for abbr in _INFORMAL_ABBR
}


def normalize_informal_abbreviations(text: str) -> str:
    """
    Normalisasi singkatan informal Bahasa Indonesia ke bentuk baku.

    Dipanggil di full_preprocess_pipeline() SEBELUM expand_abbreviations()
    agar singkatan yang sudah formal tidak salah terdeteksi sebagai akronim.

    Replace ALL kemunculan (bukan hanya pertama) karena singkatan informal
    harus konsisten di seluruh teks — berbeda dengan akronim kapital yang
    hanya di-expand sekali untuk menghindari redundansi.
    """
    if not text:
        return text
    for abbr, expansion in _INFORMAL_ABBR.items():
        pattern = _INFORMAL_ABBR_RE[abbr]
        text = pattern.sub(expansion, text)
    return text


# ════════════════════════════════════════════════════════════════════════════
# NOVELTY 1 — NLP-Based Abbreviation Expansion (akronim kapital)
# ════════════════════════════════════════════════════════════════════════════
#
# MENGAPA INI NOVELTY:
#   Pendekatan hardcoded hanya bisa handle singkatan yang sudah diketahui
#   saat coding. Pendekatan baru ini:
#   (a) Deteksi otomatis pola singkatan kapital via regex NLP
#   (b) Cari ekspansi dari Wikipedia API (zero hardcode)
#   (c) Fallback ke inline definition jika ada
#   (d) Cache hasil lookup dengan lru_cache
#   (e) Circuit breaker: maks 20 Wikipedia lookup per dokumen
#       → mencegah pipeline lambat pada dokumen panjang

_WIKIPEDIA_API     = "https://id.wikipedia.org/w/api.php"
_WIKIPEDIA_TIMEOUT = 3   # detik
_MAX_WIKI_LOOKUPS  = 20  # circuit breaker: maks lookup per dokumen


@lru_cache(maxsize=512)
def _lookup_wikipedia(abbr: str) -> Optional[str]:
    """
    Cari ekspansi singkatan via Wikipedia API (Bahasa Indonesia).
    Hasil di-cache per session agar tidak query berulang.

    Threshold match_ratio dinaikkan ke 0.75 (dari 0.6) untuk mengurangi
    false positive — minimal 3 dari 4 huruf singkatan harus ada di judul.
    """
    try:
        params = {
            "action"   : "opensearch",
            "search"   : abbr,
            "limit"    : 3,
            "namespace": 0,
            "format"   : "json",
        }
        resp = requests.get(_WIKIPEDIA_API, params=params,
                            timeout=_WIKIPEDIA_TIMEOUT)
        if resp.status_code != 200:
            return None

        data   = resp.json()
        titles = data[1] if len(data) > 1 else []

        for title in titles:
            title_upper = title.upper()
            abbr_chars  = [c for c in abbr if c.isalpha()]
            matches     = sum(1 for ch in abbr_chars if ch in title_upper)
            match_ratio = matches / len(abbr_chars) if abbr_chars else 0

            # Threshold 0.75 — lebih ketat dari versi lama (0.6)
            if match_ratio >= 0.75 and len(title.split()) >= 2:
                return f"{title} ({abbr})"

        return None

    except Exception:
        return None


def _extract_inline_definition(text: str, abbr: str) -> Optional[str]:
    """
    Cari definisi singkatan yang sudah ada di teks itu sendiri.
    Pola: "Central Processing Unit (CPU)" atau "CPU (Central Processing Unit)"
    Prioritas tertinggi karena penulis dokumen sendiri yang mendefinisikan.
    """
    pattern_before = re.compile(
        r"((?:[A-Z][a-z]+\s+){1,5})\(" + re.escape(abbr) + r"\)"
    )
    pattern_after = re.compile(
        re.escape(abbr) + r"\s*\(((?:[A-Z][a-z]+\s*){1,5})\)"
    )

    m = pattern_before.search(text)
    if m:
        expansion = m.group(1).strip()
        if len(expansion.split()) >= 2:
            return f"{expansion} ({abbr})"

    m = pattern_after.search(text)
    if m:
        expansion = m.group(1).strip()
        if len(expansion.split()) >= 2:
            return f"{expansion} ({abbr})"

    return None


def expand_abbreviations(text: str) -> str:
    """
    Expand singkatan kapital secara otomatis dengan 3 strategi bertingkat:

    Strategi 1 — Inline definition (prioritas tertinggi)
    Strategi 2 — Wikipedia API lookup (dengan circuit breaker maks 20 lookup)
    Strategi 3 — Skip (lebih baik tidak expand daripada expand salah)

    Hanya kemunculan PERTAMA yang di-expand agar tidak redundan.
    """
    candidates        = set(ABBR_DETECT_RE.findall(text))
    expanded_abbrs    = set()
    wiki_lookup_count = 0  # counter circuit breaker

    for abbr in candidates:
        if abbr in expanded_abbrs:
            continue

        expansion = _extract_inline_definition(text, abbr)

        if expansion is None:
            # Circuit breaker — jangan query Wikipedia terlalu banyak
            if wiki_lookup_count >= _MAX_WIKI_LOOKUPS:
                logger.debug(f"Wikipedia lookup limit ({_MAX_WIKI_LOOKUPS}) tercapai, skip {abbr}")
                continue
            expansion = _lookup_wikipedia(abbr)
            wiki_lookup_count += 1

        if expansion is None:
            continue

        pattern = re.compile(r"\b" + re.escape(abbr) + r"\b")
        new_text, n_replaced = pattern.subn(expansion, text, count=1)
        if n_replaced > 0:
            text = new_text
            expanded_abbrs.add(abbr)
            logger.debug(f"Expanded: {abbr} → {expansion}")

    return text


# ════════════════════════════════════════════════════════════════════════════
# NOVELTY 2 — Heading-to-Sentence Conversion
# ════════════════════════════════════════════════════════════════════════════

def heading_to_sentence(text: str) -> str:
    """Konversi heading/judul pendek menjadi kalimat pengantar."""
    stripped = text.strip()
    if COMPLETE_RE.search(stripped):
        return text
    words = stripped.split()
    if len(words) > 8:
        return text
    cleaned = STRIP_NUM_RE.sub("", stripped).strip()
    if not cleaned:
        return text
    return f"Berikut adalah penjelasan tentang {cleaned.title()}."


# ════════════════════════════════════════════════════════════════════════════
# NOVELTY 3 — Post-Reconstruction Quality Filter
# ════════════════════════════════════════════════════════════════════════════

def filter_reconstructed_sentences(text: str, min_words: int = 6) -> str:
    """Buang kalimat terlalu pendek hasil rekonstruksi."""
    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = SENT_SPLIT.split(text)
    filtered = [s for s in sents if len(s.split()) >= min_words]
    return " ".join(filtered) if filtered else text


# ════════════════════════════════════════════════════════════════════════════
# NOVELTY 4 — Section Context Stitching
# ════════════════════════════════════════════════════════════════════════════

def attach_section_context(chunks: List[Dict],
                           max_prefix_words: int = 12) -> List[Dict]:
    """Tambahkan prefix konteks section ke setiap chunk."""
    current_section = ""
    enriched        = []
    for ch in chunks:
        text  = ch.get("text", "")
        words = text.split()
        if len(words) <= max_prefix_words and not COMPLETE_RE.search(text.strip()):
            current_section = text.strip()
            enriched.append(ch)
            continue
        ch = dict(ch)
        if current_section:
            section_short = " ".join(current_section.split()[:max_prefix_words])
            if not text.startswith(section_short[:15]):
                ch["text"] = f"{section_short}. {text}"
        ch["section"] = current_section
        enriched.append(ch)
    return enriched


# ════════════════════════════════════════════════════════════════════════════
# FUNGSI INTI PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Noise removal, normalisasi unicode, dehyphenation, line joining."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = CTRL_CHAR_RE.sub(" ", text)
    text = re.sub(r"\.{3,}", " ", text)
    text = DEHYPHEN_RE.sub(r"\1\2", text)
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not any(p.search(line) for p in _NOISE_RE):
            clean_lines.append(line)
    joined = []
    for line in clean_lines:
        if not joined:
            joined.append(line)
            continue
        prev = joined[-1].strip()
        if not prev:
            joined.append(line)
            continue
        if prev[-1] in ".!?" and (line[0].isupper() or line[0].isdigit()
                                   or line.startswith(("•", "-", "*"))):
            joined.append(line)
        elif prev[-1] not in ".!?:" and (
            line[0].islower()
            or (line.split()[0].lower() in CONJUNCTIONS if line.split() else False)
        ):
            joined[-1] = joined[-1] + " " + line
        else:
            joined[-1] = joined[-1] + "\n" + line
    text = "\n\n".join(joined)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def is_toc_block(text: str) -> bool:
    """Deteksi apakah teks adalah blok Daftar Isi."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    if _STRUCTURAL_HEADER_RE.match(lines[0].lower()):
        return True
    toc_lines = sum(1 for l in lines if _TOC_LINE_RE.match(l))
    if len(lines) > 0 and toc_lines / len(lines) > 0.40:
        return True
    return False


def is_content_page(text: str) -> bool:
    """Filter halaman non-konten: Kata Pengantar, Daftar Isi, ISBN, colophon."""
    skip_kw = ("kata pengantar", "daftar isi", "isbn", "hak cipta",
               "cetakan", "diterbitkan", "perancang sampul", "ikapi")
    low = text.lower()
    if len(text.split()) < 20:
        return False
    if sum(1 for kw in skip_kw if kw in low) >= 2:
        return False
    if is_toc_block(text):
        return False
    if _PUBLICATION_META_RE.search(low):
        return False
    return True


def is_educational_content(text: str) -> bool:
    """Filter konten non-edukatif per paragraf."""
    if not text or not text.strip():
        return False
    low   = text.lower().strip()
    words = low.split()
    if not words:
        return False
    if words[0] in _TASK_KEYWORDS:
        return False
    lines_lower = [l.strip() for l in low.split("\n") if l.strip()]
    if lines_lower and lines_lower[0] in _TASK_KEYWORDS:
        return False
    if len(set(words) & _TASK_KEYWORDS) >= 3:
        return False
    stripped_low = text.strip()
    if _STRUCTURAL_HEADER_RE.match(stripped_low.lower()):
        return False
    if _COLOPHON_RE.match(stripped_low):
        return False
    if _PUBLICATION_META_RE.search(low):
        return False
    if _TOC_LINE_RE.match(stripped_low):
        return False
    if is_toc_block(text):
        return False
    if _TOC_INLINE_RE.search(text):
        return False
    return True


def merge_short_paragraphs(paras: List[str], min_words: int = 30) -> List[str]:
    if not paras:
        return paras
    merged = []
    for p in paras:
        if merged and len(merged[-1].split()) < min_words:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    if len(merged) > 1 and len(merged[-1].split()) < min_words:
        merged[-2] = merged[-2] + " " + merged[-1]
        merged.pop()
    return merged


def classify_paragraph(text: str) -> Dict:
    lines  = [l.strip() for l in text.split("\n") if l.strip()]
    total  = max(len(lines), 1)
    n_bul  = sum(1 for l in lines if BULLET_RE.match(l))
    n_num  = sum(1 for l in lines if NUMBER_RE.match(l))
    n_comp = sum(1 for l in lines if COMPLETE_RE.search(l))
    if   n_bul / total > 0.5:  ptype = "bullet"
    elif n_num / total > 0.4:  ptype = "numbered_list"
    elif n_comp / total > 0.6: ptype = "structured"
    elif len(lines) <= 2:      ptype = "heading"
    else:                      ptype = "mixed"
    return {"type": ptype, "has_bullet": n_bul > 0, "has_numbered": n_num > 0,
            "word_count": len(text.split()), "lines": lines}


def reconstruct_numbered(text: str) -> str:
    lines  = text.split("\n")
    result = []
    cur_num, cur_item = "", ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = NUMBER_CAP_RE.match(line)
        if m:
            if cur_item:
                content = cur_item.strip().rstrip(".")
                ordinal = ORDINALS.get(cur_num, f"Poin {cur_num}")
                result.append(f"{ordinal}, {content[0].lower()+content[1:]}.")
            cur_num, cur_item = m.group(1), m.group(2).strip()
        else:
            cur_item += " " + line if cur_item else line
    if cur_item:
        content = cur_item.strip().rstrip(".")
        ordinal = ORDINALS.get(cur_num, f"Poin {cur_num}")
        result.append(f"{ordinal}, {content[0].lower()+content[1:]}.")
    return " ".join(result)


def reconstruct_bullets(text: str) -> str:
    lines = text.split("\n")
    items, prefix = [], ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = BULLET_CAP_RE.match(line)
        if m:
            items.append(m.group(1).strip().rstrip("."))
        else:
            prefix += line + " "
    if not items:
        return text
    if len(items) == 1:
        joined = items[0]
    elif len(items) == 2:
        joined = f"{items[0]} dan {items[1]}"
    else:
        joined = ", ".join(items[:-1]) + f", dan {items[-1]}"
    prefix = prefix.strip()
    return (f"{prefix} Hal-hal tersebut meliputi: {joined}." if prefix
            else f"Terdapat beberapa hal, yaitu: {joined}.")


def reconstruct_paragraph(text: str, info: Dict) -> str:
    ptype = info["type"]
    if ptype == "heading":
        return heading_to_sentence(text)
    if ptype == "numbered_list":
        return reconstruct_numbered(text)
    elif ptype == "bullet":
        return reconstruct_bullets(text)
    elif ptype == "mixed":
        if info["has_numbered"]: return reconstruct_numbered(text)
        if info["has_bullet"]:   return reconstruct_bullets(text)
    return text


def restore_punctuation(text: str) -> str:
    if not text or not text.strip():
        return text
    text = PUNCT_AFTER_RE.sub(r"\1 \2", text)
    text = PUNCT_BEFORE_RE.sub(r"\1", text)
    lines  = text.split("\n")
    merged = []
    i      = 0
    while i < len(lines):
        cur = lines[i].strip()
        if not cur:
            merged.append("")
            i += 1
            continue
        while (i + 1 < len(lines)
               and cur and cur[-1] not in ".!?:"
               and lines[i+1].strip()
               and not NEXT_LINE_RE.match(lines[i+1].strip())):
            i += 1
            nxt = lines[i].strip()
            if nxt and (nxt[0].islower()
                        or nxt.split()[0].lower() in CONJUNCTIONS):
                cur = cur + " " + nxt
            else:
                merged.append(cur)
                cur = nxt
                break
        merged.append(cur)
        i += 1
    text = "\n".join(merged)
    paras = []
    for p in text.split("\n\n"):
        p = p.strip()
        if p and p[-1] not in ".!?" and len(p.split()) > 5:
            p += "."
        paras.append(p)
    text = "\n\n".join(paras)
    text = CAPITALIZE_RE.sub(
        lambda m: m.group(0)[:-1] + m.group(0)[-1].upper(), text
    )
    return MULTI_SPACE_RE.sub(" ", text).strip()


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


# ════════════════════════════════════════════════════════════════════════════
# SEMANTIC CHUNKING — LlamaIndex + paraphrase-multilingual-MiniLM-L12-v2
# ════════════════════════════════════════════════════════════════════════════

_EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_splitter = None


def _get_splitter():
    """
    Lazy-load SemanticSplitterNodeParser.

    breakpoint_percentile_threshold=75:
        Potong chunk jika similarity drop ke percentile 75.
        Nilai lebih rendah → chunk lebih kecil, lebih fokus per topik.
        Mencegah satu chunk berisi beberapa konteks berbeda.

    buffer_size=1:
        Lihat 1 kalimat di kiri/kanan sebagai konteks.
    """
    global _splitter
    if _splitter is not None:
        return _splitter
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        logger.info(f"Loading embedding model: {_EMBED_MODEL_NAME}")
        embed_model = HuggingFaceEmbedding(model_name=_EMBED_MODEL_NAME)
        _splitter   = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=1,
            breakpoint_percentile_threshold=50,  # lebih kecil = chunk lebih fokus
        )
        logger.info("Semantic splitter ready.")
    except ImportError:
        logger.warning(
            "llama-index tidak terinstall. Fallback ke sentence-aware chunking.\n"
            "Install: pip install llama-index llama-index-embeddings-huggingface sentence-transformers"
        )
        _splitter = None
    return _splitter


def _sentence_aware_chunk(text: str, max_tok: int = 400,
                           overlap: int = 50, meta: Dict = None) -> List[Dict]:
    """Fallback chunking berbasis jumlah token + overlap kalimat."""
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return []
    chunks, cid, start = [], 0, 0
    while start < len(sents):
        curr_sents, curr_tok, end = [], 0, start
        while end < len(sents):
            stok = estimate_tokens(sents[end])
            if curr_tok + stok > max_tok and curr_sents:
                break
            curr_sents.append(sents[end])
            curr_tok += stok
            end      += 1
        body = " ".join(curr_sents)
        if body.strip():
            entry = {
                "chunk_id"  : cid,
                "text"      : body,
                "n_tokens"  : curr_tok,
                "n_sents"   : len(curr_sents),
                "word_count": len(body.split()),
                "method"    : "sentence_aware",
            }
            if meta:
                entry.update(meta)
            chunks.append(entry)
            cid += 1
        ov_tok, ov_cnt = 0, 0
        for s in reversed(curr_sents):
            ov_tok += estimate_tokens(s)
            ov_cnt += 1
            if ov_tok >= overlap:
                break
        start = max(end - ov_cnt, start + 1)
    return chunks


def _split_large_chunk(body: str, max_words: int = 120) -> List[str]:
    """
    Pecah chunk yang terlalu besar menjadi sub-chunk berbasis kalimat.
    Setiap sub-chunk maksimal max_words kata.
    Mencegah 1 chunk berisi banyak konteks berbeda.
    """
    try:
        sents = nltk.sent_tokenize(body)
    except Exception:
        sents = SENT_SPLIT.split(body)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return [body]

    sub_chunks, current, cur_words = [], [], 0
    for s in sents:
        sw = len(s.split())
        if current and cur_words + sw > max_words:
            sub_chunks.append(" ".join(current))
            current, cur_words = [s], sw
        else:
            current.append(s)
            cur_words += sw
    if current:
        sub_chunks.append(" ".join(current))
    return sub_chunks if sub_chunks else [body]


def chunk_text(text: str, max_tok: int = 300, overlap: int = 30,
               meta: Dict = None) -> List[Dict]:
    """
    Semantic chunking utama via LlamaIndex.
    Fallback otomatis ke sentence-aware jika LlamaIndex tidak tersedia.
    Chunk besar (>120 kata) dipecah lagi agar setiap chunk fokus 1 topik.
    """
    if not text or not text.strip():
        return []
    splitter = _get_splitter()
    raw_chunks = []
    if splitter is not None:
        try:
            from llama_index.core import Document as LlamaDocument
            doc   = LlamaDocument(text=text)
            nodes = splitter.get_nodes_from_documents([doc])
            for i, node in enumerate(nodes):
                body = node.get_content().strip()
                if not body or len(body.split()) < 10:
                    continue
                raw_chunks.append((i, body))
            if raw_chunks:
                logger.info(f"Semantic chunking: {len(raw_chunks)} raw chunks")
            else:
                logger.warning("Semantic splitter menghasilkan 0 chunk, fallback.")
        except Exception as e:
            logger.warning(f"Semantic chunking error ({e}), fallback.")

    if not raw_chunks:
        logger.info("Menggunakan sentence-aware chunking (fallback).")
        return _sentence_aware_chunk(text, max_tok, overlap, meta)

    # Pecah chunk besar menjadi sub-chunk (maks 120 kata per chunk)
    chunks = []
    cid = 0
    for orig_i, body in raw_chunks:
        n_words = len(body.split())
        if n_words > 120:
            sub = _split_large_chunk(body, max_words=100)
        else:
            sub = [body]
        for s_body in sub:
            s_words = len(s_body.split())
            if s_words < 10:
                continue
            entry = {
                "chunk_id"  : cid,
                "text"      : s_body,
                "n_tokens"  : estimate_tokens(s_body),
                "n_sents"   : s_body.count(".") + s_body.count("?") + s_body.count("!"),
                "word_count": s_words,
                "method"    : "semantic",
            }
            if meta:
                entry.update(meta)
            chunks.append(entry)
            cid += 1
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — 9 Tahap
# ════════════════════════════════════════════════════════════════════════════

def full_preprocess_pipeline(raw_text: str) -> Dict:
    """
    Pipeline preprocessing 9 tahap untuk dokumen akademik Bahasa Indonesia.

    Tahap 1 — clean_text()
        Noise removal, normalisasi unicode NFKC, dehyphenation PDF,
        line joining untuk kalimat yang terpotong pindah baris.

    Tahap 2 — Pre-filter halaman non-konten (per-paragraf)
        Buang Kata Pengantar, Daftar Isi, colophon, ISBN.

    Tahap 3 — Filter konten non-edukatif per-paragraf
        Buang instruksi tugas/latihan, baris TOC inline, metadata publikasi.

    Tahap 4 — merge_short_paragraphs()
        Gabungkan paragraf < 30 kata.

    Tahap 5 — reconstruct_paragraph() [termasuk heading_to_sentence()]
        bullet/numbered list → kalimat naratif.
        heading → kalimat pengantar (novelty).

    Tahap 6 — filter_reconstructed_sentences() [novelty]
        Buang fragmen pendek (< 6 kata) sisa rekonstruksi.

    Tahap 6b — normalize_informal_abbreviations() [novelty baru]
        Normalisasi singkatan informal BI (yg→yang, dgn→dengan, tsb→tersebut, dll.)
        sebelum Wikipedia lookup agar teks sudah bersih saat di-expand.

    Tahap 7 — expand_abbreviations() [novelty utama]
        Deteksi akronim kapital via regex, expand via Wikipedia API.
        Circuit breaker maks 20 lookup. Tidak hardcoded.

    Tahap 8 — restore_punctuation()
        Tambah/perbaiki tanda baca, kapitalisasi awal kalimat.

    Tahap 9 — semantic chunk_text() + attach_section_context() [novelty]
        Semantic chunking berbasis cosine similarity embedding kalimat.
        Setiap chunk diberi prefix section context.
        Fallback otomatis ke sentence-aware chunking.

    Returns:
        Dict: raw, cleaned, reconstructed, expanded, restored, chunks,
              pipeline_stages
    """
    # Tahap 1
    cleaned = clean_text(raw_text)

    # Tahap 2 — filter non-konten per-paragraf
    paras = [p.strip() for p in PARA_SPLIT_RE.split(cleaned) if p.strip()]
    filtered_paras, in_task_block = [], False
    for p in paras:
        if not is_educational_content(p):
            in_task_block = True
            continue
        if in_task_block and (p.startswith(("•", "-", "*"))
                               or (len(p) > 1 and p[0].isdigit() and p[1] == ".")):
            continue
        in_task_block = False
        filtered_paras.append(p)

    # Tahap 3 — gabungkan paragraf pendek
    paras = merge_short_paragraphs(filtered_paras, min_words=30)

    # Tahap 4 — rekonstruksi + heading → kalimat
    recon_paras = [reconstruct_paragraph(p, classify_paragraph(p)) for p in paras]
    recon       = "\n\n".join(recon_paras)

    # Tahap 5 — filter fragmen pendek
    filtered_recon = []
    for p in recon.split("\n\n"):
        p = p.strip()
        if p:
            filtered_recon.append(filter_reconstructed_sentences(p, min_words=6))
    recon = "\n\n".join(p for p in filtered_recon if p.strip())

    # Tahap 6b — normalisasi singkatan informal (BARU)
    # Dikerjakan SEBELUM expand_abbreviations agar teks sudah formal
    recon = normalize_informal_abbreviations(recon)

    # Tahap 7 — expand akronim kapital (NLP-based + Wikipedia)
    expanded = expand_abbreviations(recon)

    # Tahap 8 — punctuation restoration
    restored = restore_punctuation(expanded)

    # Tahap 9 — semantic chunking + section context
    chunks = chunk_text(restored)
    chunks = attach_section_context(chunks)

    # Post-filter chunks: safety net untuk noise yang lolos
    def _chunk_is_clean(ch: Dict) -> bool:
        txt = ch.get("text", "")
        if not txt.strip():
            return False
        if is_toc_block(txt):
            return False
        lines     = [l.strip() for l in txt.split("\n") if l.strip()]
        toc_count = sum(1 for l in lines if _TOC_LINE_RE.match(l))
        if lines and toc_count / len(lines) > 0.3:
            return False
        if len(txt.split()) < 15 and (
            _COLOPHON_RE.match(txt.strip())
            or _STRUCTURAL_HEADER_RE.match(txt.strip().lower())
            or _PUBLICATION_META_RE.search(txt.lower())
        ):
            return False
        return True

    chunks = [ch for ch in chunks if _chunk_is_clean(ch)]

    pipeline_stages = [
        {"stage": "1. Raw Input",              "text": raw_text},
        {"stage": "2. Cleaned",                "text": cleaned},
        {"stage": "3. Reconstructed",          "text": recon},
        {"stage": "4. Informal Abbr Norm.",    "text": recon},
        {"stage": "5. Abbreviation Expanded",  "text": expanded},
        {"stage": "6. Punct. Restored",        "text": restored},
        {"stage": "7. Chunking Method",        "text": chunks[0].get("method", "sentence_aware") if chunks else "no chunks"},
    ]

    return {
        "raw"            : raw_text,
        "cleaned"        : cleaned,
        "reconstructed"  : recon,
        "expanded"       : expanded,
        "restored"       : restored,
        "chunks"         : chunks,
        "pipeline_stages": pipeline_stages,
    }


# ════════════════════════════════════════════════════════════════════════════
# ANSWER SELECTION
# ════════════════════════════════════════════════════════════════════════════

_BAD_ANSWER_STARTS = re.compile(
    r"^(bab|gambar|tabel|pertama|kedua|ketiga|keempat|kelima|keenam|"
    r"ketujuh|kedelapan|berikut|adalah|kemudian|selain|adapun|oleh|"
    r"berikut adalah penjelasan tentang|cara kerjanya|cara kerja|"
    r"rumus|contoh|misalnya|perhatikan|lihat|amati|pilih|lakukan|"
    r"klik|tekan|ketik|buka|tutup|masuk|keluar|tambah|hapus|ubah|"
    r"langkah|tahap|proses|prosedur)",
    re.IGNORECASE)
_DIGIT_START_RE = re.compile(r"^\d+")

# Pola definisi eksplisit: "X adalah/merupakan/yaitu Y"
# Subjek (X) 3-80 karakter, diikuti kata definisi, lalu penjelasan ≥10 karakter
_DEFINITION_RE = re.compile(
    r"^.{3,80}\s+(adalah|merupakan|yaitu|ialah|didefinisikan\s+sebagai"
    r"|dapat\s+diartikan\s+sebagai|dikenal\s+sebagai)\s+.{10,}",
    re.IGNORECASE
)

# Pola karakteristik: "X memiliki/berfungsi/terdiri dari..."
_CHARACTERISTIC_RE = re.compile(
    r"^.{3,60}\s+(memiliki|berfungsi|digunakan\s+untuk|bertujuan\s+untuk"
    r"|terdiri\s+(atas|dari)|meliputi|mencakup|menghasilkan)\s+.{10,}",
    re.IGNORECASE
)

# Pola proses/cara kerja: "X bekerja dengan/dilakukan dengan..."
_PROCESS_RE = re.compile(
    r"^.{3,60}\s+(bekerja\s+dengan|dilakukan\s+dengan|diterapkan\s+dengan"
    r"|dijalankan\s+dengan|diproses\s+(melalui|dengan))\s+.{10,}",
    re.IGNORECASE
)

# Pola kalimat rumus/matematika — bukan jawaban yang baik untuk kuis
_FORMULA_RE = re.compile(
    r"[A-Za-z]\s*\([a-zA-Z,\s]+\)\s*[=→←]"
    r"|[=→←≤≥≠]\s*\d"
    r"|\b\d+\s*[+\-\*/]\s*\d+"
)

# Pola kalimat dekoratif dengan banyak emoji
_EMOJI_HEAVY_RE = re.compile(
    r"[\U0001F300-\U0001FAFF].*[\U0001F300-\U0001FAFF]"
)


def _answer_score(sent: str) -> float:
    """
    Skor kualitas kalimat sebagai pseudo-answer.

    Tier scoring (tinggi ke rendah):
    1. Kalimat definisi eksplisit "X adalah/merupakan/yaitu Y" (lengkap) → +10
    2. Kalimat karakteristik "X memiliki/berfungsi/terdiri dari" → +5
    3. Kalimat proses "X dilakukan dengan/bekerja melalui" → +3
    4. Ada kata definisi tapi bukan predikat utama → +1.5
    5. Kalimat instruksional/rumus/emoji → -1.0

    Hanya kalimat dengan SUBYEK JELAS + PREDIKAT DEFINISI yang mendapat skor tinggi.
    """
    words = sent.split()
    n     = len(words)

    if n < 6:
        return -1.0
    if _BAD_ANSWER_STARTS.match(sent) or _DIGIT_START_RE.match(sent):
        return -1.0
    if _FORMULA_RE.search(sent):
        return -1.0
    if _EMOJI_HEAVY_RE.search(sent):
        return -1.0

    score = 0.0

    if _DEFINITION_RE.match(sent):
        # Pastikan bagian setelah predikat cukup panjang (bukan definisi setengah-setengah)
        score += 10.0
    elif _CHARACTERISTIC_RE.match(sent):
        score += 5.0
    elif _PROCESS_RE.match(sent):
        score += 3.0
    elif any(kw in words for kw in ("adalah", "merupakan", "yaitu", "ialah")):
        score += 1.5

    if 10 <= n <= 35:
        score += 2.0
    elif 6 <= n < 10:
        score += 0.5
    elif n > 45:
        score -= 1.5  # penalti lebih besar untuk jawaban terlalu panjang

    if re.search(r"\([A-Z]{2,}\)", sent):
        score += 1.5

    if sent.rstrip()[-1] in ".!?":
        score += 0.5

    return score


# ── Kata definisi yang sering jadi predikat kalimat definisi ─────────────────
_DEF_PREDICATES = (
    "adalah", "merupakan", "yaitu", "ialah",
    "didefinisikan sebagai", "dapat diartikan sebagai", "dikenal sebagai"
)

# ── Kata ganti / sinonim yang bisa dipakai untuk mengecoh ────────────────────
_DEF_REPLACEMENTS = [
    ("adalah",                  ["bukan", "bukanlah", "tidak termasuk", "berbeda dari"]),
    ("merupakan",               ["bukan merupakan", "tidak merupakan", "berlawanan dengan"]),
    ("yaitu",                   ["bukan", "tidak sama dengan"]),
    ("ialah",                   ["bukanlah", "bukan"]),
    ("didefinisikan sebagai",   ["tidak dapat didefinisikan sebagai", "bukan didefinisikan sebagai"]),
    ("dapat diartikan sebagai", ["tidak dapat diartikan sebagai"]),
    ("dikenal sebagai",         ["tidak dikenal sebagai"]),
]


def _get_definition_parts(sent: str):
    """
    Pisahkan kalimat definisi menjadi (subyek, predikat, keterangan).
    Return None jika bukan kalimat definisi.
    """
    for pred in _DEF_PREDICATES:
        pattern = re.compile(
            r"^(.{3,80})\s+(" + re.escape(pred) + r")\s+(.{10,})",
            re.IGNORECASE
        )
        m = pattern.match(sent)
        if m:
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return None


def generate_misleading_distractors(correct: str, chunk_text: str = "", n: int = 3) -> list:
    """
    Buat distractor yang MIRIP tapi SALAH dari jawaban benar.
    Strategi:
    1. Tukar bagian keterangan definisi dengan keterangan dari kalimat lain di chunk
    2. Negasi predikat ("adalah" → "bukan")
    3. Tukar subyek antar kalimat definisi dalam chunk
    """
    distractors = []
    parts = _get_definition_parts(correct)

    # Kumpulkan kalimat definisi lain dari chunk sebagai bahan distractor
    other_defs = []
    if chunk_text:
        try:
            sents = nltk.sent_tokenize(chunk_text)
        except Exception:
            sents = SENT_SPLIT.split(chunk_text)
        for s in sents:
            s = s.strip()
            if not s or s.lower() == correct.lower():
                continue
            p = _get_definition_parts(s)
            if p:
                other_defs.append(p)  # (subj, pred, ket)

    if parts:
        subj, pred, ket = parts

        # Strategi 1: Tukar keterangan dengan keterangan definisi lain dari chunk
        used_kets = set()
        for other_subj, other_pred, other_ket in other_defs:
            if other_ket.lower() != ket.lower() and other_ket not in used_kets:
                d = f"{subj} {pred} {other_ket}"
                if d.lower() != correct.lower() and len(d.split()) >= 5:
                    distractors.append(d)
                    used_kets.add(other_ket)
                if len(distractors) >= n:
                    break

        # Strategi 2: Tukar subyek dengan subyek definisi lain
        if len(distractors) < n:
            for other_subj, other_pred, other_ket in other_defs:
                if other_subj.lower() != subj.lower():
                    d = f"{other_subj} {pred} {ket}"
                    if d.lower() != correct.lower() and d not in distractors and len(d.split()) >= 5:
                        distractors.append(d)
                if len(distractors) >= n:
                    break

        # Strategi 3: Modifikasi keterangan (potong / tambah frase negatif)
        if len(distractors) < n:
            ket_words = ket.split()
            # Variasi: ambil setengah awal keterangan + frase lain
            if len(ket_words) >= 6:
                half_ket = " ".join(ket_words[:len(ket_words)//2])
                # Tambah keterangan dari kalimat lain untuk melengkapi
                for other_subj, other_pred, other_ket in other_defs:
                    other_ket_words = other_ket.split()
                    if len(other_ket_words) >= 3:
                        suffix = " ".join(other_ket_words[-3:])
                        d = f"{subj} {pred} {half_ket} {suffix}"
                        if d.lower() != correct.lower() and d not in distractors and len(d.split()) >= 5:
                            distractors.append(d)
                    if len(distractors) >= n:
                        break

    # Fallback: kalimat mirip dari chunk (overlap kata tinggi)
    if len(distractors) < n and chunk_text:
        try:
            sents = nltk.sent_tokenize(chunk_text)
        except Exception:
            sents = SENT_SPLIT.split(chunk_text)
        correct_words = set(correct.lower().split())
        candidates = []
        for s in sents:
            s = s.strip()
            if not s or s.lower() == correct.lower():
                continue
            if correct.lower()[:30] in s.lower():
                continue
            words = s.split()
            if len(words) > 35:
                s = ' '.join(words[:25])
            if not _is_clean_distractor(s):
                continue
            overlap = len(set(s.lower().split()) & correct_words)
            candidates.append((s, overlap))
        candidates.sort(key=lambda x: (x[1], len(x[0].split())), reverse=True)
        for c, _ in candidates:
            if c not in distractors:
                distractors.append(c)
            if len(distractors) >= n:
                break

    return distractors[:n]


def _is_clean_distractor(text: str) -> bool:
    """Validasi teks: minimal 3 kata, maks 35 kata, bebas noise struktural."""
    t     = text.strip()
    words = t.split()
    if not t or len(words) < 3:
        return False
    if len(words) > 35:
        return False
    if _TOC_LINE_RE.match(t):
        return False
    if _COLOPHON_RE.match(t):
        return False
    if _PUBLICATION_META_RE.search(t.lower()):
        return False
    structural_kw = [
        "daftar isi", "kata pengantar", "daftar pustaka",
        "daftar gambar", "daftar tabel", "isbn", "hak cipta",
        "bagian i", "bagian ii", "bagian iii", "bagian iv", "bagian v",
        "program studi", "mata kuliah", "disusun oleh", "tim pengampu",
    ]
    if any(kw in t.lower() for kw in structural_kw):
        return False
    return True


def pick_answer(text: str) -> str:
    """
    Pilih kalimat terbaik dari chunk sebagai pseudo-answer.

    Urutan prioritas:
    1. Kalimat definisi eksplisit LENGKAP (subyek + predikat definisi + keterangan ≥10 kata)
    2. Kalimat karakteristik/proses (skor ≥ 3)
    3. Kalimat informatif umum (skor > 0)
    4. Kalimat 10-30 kata yang tidak instruksional

    Mengembalikan None kalau tidak ada kalimat layak.
    """
    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = SENT_SPLIT.split(text)

    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return None

    scored = [(s, _answer_score(s)) for s in sents]

    # Tier 1: definisi eksplisit LENGKAP (skor >= 10)
    tier1 = [(s, sc) for s, sc in scored if sc >= 10.0]
    if tier1:
        # Pilih yang paling pendek di tier 1 agar mudah dijadikan jawaban
        return min(tier1, key=lambda x: len(x[0].split()))[0]

    # Tier 2: karakteristik atau proses (skor >= 3)
    tier2 = [(s, sc) for s, sc in scored if sc >= 3.0]
    if tier2:
        return max(tier2, key=lambda x: x[1])[0]

    # Tier 3: skor positif apapun
    tier3 = [(s, sc) for s, sc in scored if sc > 0]
    if tier3:
        return max(tier3, key=lambda x: x[1])[0]

    # Tidak ada kalimat layak
    return None