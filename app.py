#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patched app.py — Gemini-first skill extraction with guaranteed JD phrase inclusion
and caching of JD skill extraction (avoid re-calling Gemini when JD unchanged).

Features:
- Calls Gemini to extract skills from JD/resume text.
- Post-processes Gemini output with _expand_and_imply (split combined tokens like "c/c++").
- Forces inclusion of explicit high-priority phrases found in the JD text (GUARANTEED_PHRASES).
- Allows the single-letter token 'c' to pass the short-token filter.
- Caches JD skill extraction in st.session_state and reuses if JD not changed.
"""

import os
import io
import re
import json
import time
import zipfile
import hashlib
from datetime import datetime
from collections import OrderedDict

import docx
import pdfplumber
import pandas as pd
import streamlit as st

# ------------------ HARD-CODED GEMINI KEY (replace locally) ------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# ------------------ genai client ------------------
try:
    from google import genai
except Exception:
    genai = None

if genai is None:
    raise RuntimeError("Install google-genai: pip install google-genai")

client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================== #
# GLOBAL DEBUG REGISTRY (for clean, structured logs)
# ============================================================== #
EXTRACT_DEBUG_REGISTRY = {}  # { filename: {...} }

# ------------------ LLM helpers (skills + years) ------------------

MAX_CHARS = 14000

def call_gemini_for_skills(resume_text: str, model: str = "gemini-2.5-flash", max_retries: int = 2):
    """
    Returns (skills_list, raw_output).
    skills_list: list[str] lowercased tokens (may be empty).
    """
    if not resume_text:
        return [], "<empty>"

    txt = resume_text if len(resume_text) <= MAX_CHARS else resume_text[:MAX_CHARS]
    prompt = f"""
You are an extractor. Given the resume or JD text below, return ONLY a single JSON object:

{{"skills": ["skill1","skill2", ...]}}

Rules:
- Return short canonical skill tokens like "python", "c++", "embedded linux", "device tree", "u-boot", "yocto", "i2c", "spi", "git".
- Normalize common variants (react.js -> react, node js -> node.js, powerbi -> power bi).
- If the text contains combined forms like "C/C++", return both "c" and "c++" as separate items.
- Deduplicate and include only skills actually present in the text.
- Do NOT include company names, addresses, or long descriptive sentences.
- Output EXACTLY one JSON object and nothing else.

Text:
\"\"\"{txt}\"\"\" 
""".strip()

    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            raw = resp.text if hasattr(resp, "text") else str(resp)
            # find JSON object
            m = re.search(r'\{.*\}', raw, flags=re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    skills = obj.get("skills", [])
                    if isinstance(skills, list):
                        # normalize & dedupe preserving order
                        cleaned = []
                        seen = set()
                        for s in skills:
                            if not isinstance(s, str):
                                continue
                            tok = s.strip().lower()
                            # small normalization
                            tok = tok.replace('react.js', 'react').replace('reactjs', 'react')
                            tok = tok.replace('node js', 'node.js').replace('nodejs', 'node.js')
                            tok = tok.replace('powerbi', 'power bi')
                            tok = re.sub(r'[\._]+', ' ', tok)
                            tok = re.sub(r'\s+', ' ', tok).strip()
                            if tok and tok not in seen:
                                seen.add(tok)
                                cleaned.append(tok)
                        return cleaned, raw
                except Exception:
                    pass

            # fallback: try to parse an array literal
            arr_match = re.search(r'\[\s*([^\]]+?)\s*\]', raw, flags=re.DOTALL)
            if arr_match:
                items = re.findall(r'["\']([^"\']+)["\']', arr_match.group(0))
                cleaned = []
                seen = set()
                for s in items:
                    tok = s.strip().lower()
                    tok = tok.replace('react.js', 'react').replace('reactjs', 'react')
                    tok = tok.replace('node js', 'node.js').replace('nodejs', 'node.js')
                    tok = re.sub(r'[\._]+', ' ', tok)
                    tok = re.sub(r'\s+', ' ', tok).strip()
                    if tok and tok not in seen:
                        seen.add(tok)
                        cleaned.append(tok)
                if cleaned:
                    return cleaned, raw

            # If we didn't get JSON, fallback to empty so caller will use local fallback
            return [], raw
        except Exception as e:
            # transient retry
            if attempt < max_retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            return [], f"<error: {e}>"

def call_gemini_for_years(resume_text: str, model: str = "gemini-2.5-flash", max_retries: int = 2):
    """
    Returns (float_years, raw_output)
    Expects LLM to return ONLY {"total_years": <float>}
    """
    if not resume_text:
        return 0.0, "<empty>"

    txt = resume_text if len(resume_text) <= MAX_CHARS else resume_text[:MAX_CHARS]
    today = datetime.now().date()
    prompt = f"""
You are a strict extractor. Given the resume text below, compute the candidate's total professional work experience in years, MERGING overlapping jobs so they are not double-counted. Return ONLY a single JSON object with this numeric field:

{{"total_years": <float>}}

Rules:
- Count full months; express years as decimal with one digit after decimal (e.g., 3.5).
- Treat "present/current" as up to today's date ({today}).
- Merge overlapping jobs before summing so overlaps are not double-counted.
- If you cannot detect any valid dates or experience, return {{"total_years": 0.0}}.
- DO NOT output any text except the single JSON object above.

Resume:
\"\"\"{txt}\"\"\" 
""".strip()

    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            raw = resp.text if hasattr(resp, "text") else str(resp)
            m = re.search(r'\{.*\}', raw, flags=re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    value = float(obj.get("total_years", 0.0))
                    return round(value, 1), raw
                except Exception:
                    pass
            # try to find numeric fallback
            m2 = re.search(r'(\d+(?:\.\d+)?)', raw)
            if m2:
                try:
                    return round(float(m2.group(1)), 1), raw
                except:
                    pass
            return 0.0, raw
        except Exception as e:
            if attempt < max_retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            return 0.0, f"<error: {e}>"

# ------------------ TEXT extraction utilities (existing) ------------------
def extract_text_from_pdf(pdf_file):
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages)
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'\(https?:\/\/\S+\)', '', text)
            return text.lower()
    except Exception:
        return ""

def extract_text_from_docx(docx_file):
    """Paragraph + table text."""
    try:
        docx_file.seek(0)
        document = docx.Document(docx_file)
        parts = []
        for p in document.paragraphs:
            if p.text:
                parts.append(p.text)
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    ct = cell.text.strip()
                    if ct:
                        parts.append(ct)
        text = "\n".join(parts)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    except Exception:
        return ""

def extract_text_from_txt(txt_file):
    try:
        txt_file.seek(0)
        raw = txt_file.getvalue()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='ignore')
        text = str(raw)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    except Exception:
        return ""

# ------------------ GEMINI-FIRST skill extraction (new) ------------------

# Minimal deterministic whitelist fallback (compact)
MINIMAL_WHITELIST = [
    "c", "c++", "python", "java", "javascript", "node.js", "react",
    "embedded linux", "yocto", "u-boot", "device tree", "i2c", "spi", "usb", "ethernet",
    "gdb", "jtag", "docker", "qemu", "firmware", "bootloader", "kernel", "kernel drivers",
    "yocto", "openwrt", "buildroot", "tpm", "wifi", "lte", "qualcomm", "mediatek", "broadcom", "nxp"
]

def _minimal_whitelist_scan(text: str):
    """
    A tiny deterministic fallback that searches MINIMAL_WHITELIST words as exact tokens.
    This is intentionally small and conservative (only used when Gemini returns nothing or errors).
    """
    if not text:
        return []
    t = text.lower()
    found = []
    for kw in MINIMAL_WHITELIST:
        pat = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pat, t):
            normalized = kw.lower()
            # normalize wifi -> wi-fi for consistency if present
            if normalized == "wifi":
                normalized = "wi-fi"
            if normalized not in found:
                found.append(normalized)
    return found

# ------------------ NEW: expansion & implied helper ------------------

def _expand_and_imply(skills_list):
    """
    Post-process raw skills list from Gemini:
    - Split combined tokens like "c/c++", "c & c++", "c and c++" into separate tokens.
    - Split on '/', '&', ',', and the word 'and' (but safely handle 'c++').
    - Normalize a few common variants (wifi -> wi-fi, drivers -> kernel drivers).
    - Add implied tokens (e.g., add 'c' if 'c++' present).
    - Preserve order while deduping.
    """
    if not skills_list:
        return []

    expanded = []
    seen = set()

    for tok in skills_list:
        if not isinstance(tok, str):
            continue
        t = tok.strip().lower()

        # Explicitly expand common compact forms before splitting
        t = t.replace('c/c++', 'c c++').replace('c & c++', 'c c++').replace('c and c++', 'c c++')
        t = t.replace('c++/c', 'c c++').replace('c/cpp', 'c c++')

        # Split on common delimiters but avoid splitting on '+' (so c++ stays intact)
        parts = re.split(r'\s*(?:/|&|,|\band\b)\s*', t)

        for p in parts:
            p = p.strip()
            if not p:
                continue

            # small normalizations
            if p == 'wifi':
                p = 'wi-fi'
            if p in ('driver', 'drivers'):
                p = 'kernel drivers'
            if p == 'kernel':
                p = 'linux kernel'

            # fix noisy tokens, remove punctuation
            p = re.sub(r'[\._]+', ' ', p)
            p = re.sub(r'\s+', ' ', p).strip()

            # skip very short or empty (but allow 'c')
            if len(p) < 2 and p != 'c':
                continue

            if p not in seen:
                seen.add(p)
                expanded.append(p)

    # implied tokens: if c++ present but c missing, add c
    if 'c++' in seen and 'c' not in seen:
        expanded.append('c')
        seen.add('c')

    return expanded

# ------------------ GUARANTEED PHRASES for JD (force-inclusion if present) ------------------

GUARANTEED_PHRASES = [
    "arm", "arm-based", "arm socs", "qualcomm", "mediatek", "broadcom", "nxp",
    "build systems", "production-ready", "production-ready images", "board diagnostics",
    "manufacturing", "manufacturing support", "hardware integration", "peripherals",
    "network interfaces", "wireless chipsets", "documentation", "open-source",
    "router platforms", "router", "firmware", "test automation", "performance optimization",
    "memory optimization", "boot optimization", "code reviews", "debugging tools"
]

def force_include_from_text(skills_list, text):
    """
    Ensure high-priority phrases present in the text are included in skills_list.
    Returns (updated_skills_list, added_list)
    """
    if not text:
        return skills_list, []

    t = text.lower()
    added = []
    # keep order: append any guaranteed phrase that's present and not already in list
    seen = set(x.lower() for x in skills_list)
    for phr in GUARANTEED_PHRASES:
        if phr in t:
            canon = phr
            # normalize a few canonical forms
            if canon == "wifi":
                canon = "wi-fi"
            if canon == "arm-based":
                canon = "arm"
            if canon.endswith("socs"):
                canon = canon.replace("socs", "socs")
            # if not present, append
            if canon not in seen:
                skills_list.append(canon)
                added.append(canon)
                seen.add(canon)
    return skills_list, added

# ------------------ CACHING HELPERS ------------------

def _text_hash(s: str) -> str:
    if not s:
        return ""
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

# Initialize session_state cache keys if not present
if "jd_hash" not in st.session_state:
    st.session_state.jd_hash = ""
if "jd_keywords_cached" not in st.session_state:
    st.session_state.jd_keywords_cached = set()

# ------------------ GEMINI-FIRST extractors (with post-processing) ------------------

def get_keywords_from_jd(jd_text: str, max_terms: int = 200):
    """
    Gemini-first extraction for JD: call LLM and return a set of skills (strings).
    Post-processes Gemini results with _expand_and_imply and force_include_from_text to ensure combined tokens are split
    and important JD phrases are included.
    """
    if not jd_text:
        return set()

    # Attempt to reuse cache if text hasn't changed
    new_hash = _text_hash(jd_text)
    if new_hash == st.session_state.jd_hash and st.session_state.jd_keywords_cached:
        # use cached
        return set(st.session_state.jd_keywords_cached)

    skills, raw = call_gemini_for_skills(jd_text)
    # Post-process expansion & implied tokens
    skills = _expand_and_imply(skills)

    # Force-include guaranteed phrases present in JD text
    skills, added = force_include_from_text(skills, jd_text)

    # Basic sanity checks on LLM response
    if skills and isinstance(skills, list):
        # Keep up to max_terms and filter out very short tokens (allow 'c')
        cleaned = []
        seen = set()
        for s in skills:
            if not isinstance(s, str):
                continue
            tok = s.strip().lower()
            tok = re.sub(r'\s+', ' ', tok)
            # allow the single-letter language 'c' (keep filtering other tiny tokens)
            if len(tok) < 2 and tok != "c":
                continue
            if tok in seen:
                continue
            seen.add(tok)
            cleaned.append(tok)
            if len(cleaned) >= max_terms:
                break

        # Cache results
        st.session_state.jd_keywords_cached = set(cleaned)
        st.session_state.jd_hash = new_hash
        return set(cleaned)

    # Fallback
    fallback = _minimal_whitelist_scan(jd_text)
    # cache fallback as well
    st.session_state.jd_keywords_cached = set(fallback)
    st.session_state.jd_hash = new_hash
    return set(fallback)

def get_skills_from_text(text: str, max_terms: int = 200):
    """
    Gemini-first extraction for resumes. Returns an ordered list (preserve LLM order).
    If Gemini fails, returns the minimal whitelist scan results as a list.
    Post-processes Gemini results with _expand_and_imply to ensure combined tokens are split.
    """
    if not text:
        return []

    skills, raw = call_gemini_for_skills(text)
    # Post-process expansion & implied tokens
    skills = _expand_and_imply(skills)

    ordered = []
    seen = set()
    if skills and isinstance(skills, list):
        for s in skills:
            if not isinstance(s, str):
                continue
            tok = s.strip().lower()
            tok = re.sub(r'\s+', ' ', tok)
            # allow the single-letter language 'c'
            if len(tok) < 2 and tok != "c":
                continue
            if tok in seen:
                continue
            seen.add(tok)
            ordered.append(tok)
            if len(ordered) >= max_terms:
                break

    if ordered:
        return ordered

    # fallback deterministic scan
    fallback = _minimal_whitelist_scan(text)
    return fallback

# -------------------------------------------------------------------------------------------------
# EXPERIENCE EXTRACTION (existing local code) - left unchanged
# -------------------------------------------------------------------------------------------------
def parse_date_any(s):
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    fmts = ("%b %Y", "%B %Y", "%Y/%m", "%m/%Y", "%Y")
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except:
            continue
    m = re.match(r'^\d{4}$', s)
    if m:
        try:
            return datetime.strptime(s, "%Y")
        except:
            pass
    return None

def extract_experience_from_resume(text, filename="", aggressive_edu=True):
    fname = filename or "unknown"
    dbg = {
        "source": "none",
        "explicit_values": [],
        "structured_intervals": [],
        "date_range_intervals": [],
        "final_years": 0.0,
    }

    print("\n" + "═"*78)
    print(f"🧾 RESUME EXPERIENCE DEBUG — {fname}")
    print("═"*78)

    if not text:
        print("⚠️ No text extracted from resume. Returning 0.0 years.")
        EXTRACT_DEBUG_REGISTRY[fname] = dbg
        return 0.0

    text_low = text.lower()

    explicit_values = []
    # 1) Explicit numeric phrases
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?|y)\b', text_low):
        try:
            val = float(m.group(1))
            explicit_values.append(val)
            print(f"🔎 Found explicit years mention: {val}")
        except:
            pass

    for m in re.finditer(r'\b(\d+)\s*[yY]\s*(\d+)\s*[mM]\b', text_low):
        try:
            val = int(m.group(1)) + int(m.group(2))/12.0
            explicit_values.append(val)
            print(f"🔎 Found Y-M pattern => {val:.2f} years")
        except:
            pass

    for m in re.finditer(r'(\d+(?:\.\d+)?)\s+years\s+of\s+experience', text_low):
        try:
            val = float(m.group(1))
            explicit_values.append(val)
            print(f"🔎 Found 'years of experience' => {val}")
        except:
            pass

    if explicit_values:
        result = round(max(explicit_values), 1)
        dbg["source"] = "explicit_number"
        dbg["explicit_values"] = explicit_values[:]
        dbg["final_years"] = result
        print(f"✅ Using explicit numeric mention → {result} years (max of {explicit_values})")
        EXTRACT_DEBUG_REGISTRY[fname] = dbg
        return result

    # 2) Section-focused parsing (Experience vs Education)
    lines = [ln.strip() for ln in re.split(r'[\r\n]+', text_low) if ln.strip()]
    current_section = None
    experience_text_lines = []

    education_keywords = re.compile(r'\b(education|academic|university|college|degree|school|certification)\b', re.I)
    experience_keywords = re.compile(r'\b(experience|employment|work history|professional experience|work experience|career)\b', re.I)

    for idx, ln in enumerate(lines):
        if experience_keywords.search(ln):
            current_section = "experience"
            print(f"📌 Section → experience (line {idx})")
            continue
        if education_keywords.search(ln):
            current_section = "education"
            print(f"📌 Section → education (line {idx})")
            continue

        if ln.isupper() and len(ln.split()) <= 4:
            if 'experience' in ln.lower() or 'employment' in ln.lower() or 'work' in ln.lower():
                current_section = "experience"
                print(f"📌 Section → EXPERIENCE (header, line {idx})")
                continue
            if 'education' in ln.lower() or 'academic' in ln.lower():
                current_section = "education"
                continue

        if aggressive_edu and current_section == "education":
            continue

        if current_section == "experience":
            experience_text_lines.append(ln)

    if experience_text_lines:
        print(f"🧱 Collected {len(experience_text_lines)} line(s) from experience section.")
    else:
        print("ℹ️ No direct experience section captured; falling back to broader scan.")

    if not experience_text_lines:
        if aggressive_edu:
            filtered_lines = []
            skip = False
            for ln in lines:
                if education_keywords.search(ln):
                    skip = True
                    continue
                if re.match(r'^[A-Z ]{2,30}$', ln) and len(ln.split()) <= 4:
                    skip = False
                if not skip:
                    filtered_lines.append(ln)
            experience_text = "\n".join(filtered_lines)
        else:
            experience_text = "\n".join(lines)
    else:
        experience_text = "\n".join(experience_text_lines)

    # 3) Structured job interval pattern
    job_exp_pattern = re.compile(
        r'(?:[\w\s]+,?\s*[\w\s]*?),\s*([\w\s]+?),\s*([A-Za-z]{3}\s*\d{4})\s*[\-–—to]{1,}\s*(present|till date|current|[A-Za-z]{3}\s*\d{4})',
        re.IGNORECASE
    )
    structured_intervals = []
    for m in job_exp_pattern.finditer(text):
        sd = parse_date_any(m.group(2))
        ed_text = m.group(3)
        ed = datetime.now() if re.search(r'(present|till date|current)', ed_text, flags=re.I) else parse_date_any(ed_text)
        if sd and ed and sd < ed:
            structured_intervals.append((sd, ed))

    if structured_intervals:
        structured_intervals.sort(key=lambda x: x[0])
        merged = []
        cur_s, cur_e = structured_intervals[0]
        for s, e in structured_intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        total_days = sum((e - s).days for s, e in merged)
        total_years = round(total_days / 365.25, 1)
        dbg["source"] = "structured_sections"
        dbg["structured_intervals"] = [(s.strftime("%b %Y"), e.strftime("%b %Y")) for s, e in merged]
        dbg["final_years"] = total_years
        print("🗂️ Structured intervals (merged):")
        for s, e in merged:
            print(f" • {s.strftime('%b %Y')} → {e.strftime('%b %Y')}")
        print(f"✅ Using structured job intervals → {total_years} years")
        EXTRACT_DEBUG_REGISTRY[fname] = dbg
        return total_years

    # 4) Generic date ranges
    intervals = []
    date_range_patterns = [
        r'([A-Za-z]{3,9}\s*\d{4})\s*[\-–—to]{1,}\s*(present|current|till date|[A-Za-z]{3,9}\s*\d{4})',
        r'(\d{4}\/\d{1,2})\s*[\-–—to]{1,}\s*(present|current|till date|\d{4}\/\d{1,2})',
        r'(\d{4})\s*[\-–—to]{1,}\s*(present|current|till date|\d{4})'
    ]
    for patt in date_range_patterns:
        for m in re.finditer(patt, experience_text, flags=re.IGNORECASE):
            sd = parse_date_any(m.group(1))
            ed = datetime.now() if re.search(r'(present|current|till date)', m.group(2), flags=re.I) else parse_date_any(m.group(2))
            if sd and ed and sd < ed:
                intervals.append((sd, ed))

    if intervals:
        intervals.sort(key=lambda x: x[0])
        merged = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        total_days = sum((e - s).days for s, e in merged)
        total_years = round(total_days / 365.25, 1)
        dbg["source"] = "date_ranges"
        dbg["date_range_intervals"] = [(s.strftime("%b %Y"), e.strftime("%b %Y")) for s, e in merged]
        dbg["final_years"] = total_years
        print("🗂️ Date ranges (merged):")
        for s, e in merged:
            print(f" • {s.strftime('%b %Y')} → {e.strftime('%b %Y')}")
        print(f"✅ Using generic date ranges → {total_years} years")
        EXTRACT_DEBUG_REGISTRY[fname] = dbg
        return total_years

    # 5) Filename hints
    if filename:
        m = re.search(r'(\d+)\s*[yY]\s*[_\-]?\s*(\d+)\s*[mM]?', filename)
        if m:
            result = round(int(m.group(1)) + int(m.group(2))/12.0, 1)
            dbg["source"] = "filename_y_m"
            dbg["final_years"] = result
            print(f"✅ Using filename Y-M pattern → {result} years (from: {filename})")
            EXTRACT_DEBUG_REGISTRY[fname] = dbg
            return result
        m = re.search(r'(\d+)\s*[yY]', filename)
        if m:
            result = float(m.group(1))
            dbg["source"] = "filename_y"
            dbg["final_years"] = result
            print(f"✅ Using filename Y pattern → {result} years (from: {filename})")
            EXTRACT_DEBUG_REGISTRY[fname] = dbg
            return result

    # 6) Simple year-year fallback
    m = re.search(r'(\d{4})\s*[\-–—to]{1,}\s*(\d{4}|present|current)', text_low)
    if m:
        sd = parse_date_any(m.group(1))
        ed = datetime.now() if re.search(r'(present|current)', m.group(2)) else parse_date_any(m.group(2))
        if sd and ed and sd < ed:
            result = round((ed - sd).days / 365.25, 1)
            dbg["source"] = "simple_year_year"
            dbg["final_years"] = result
            print(f"✅ Using simple year-year match → {result} years")
            EXTRACT_DEBUG_REGISTRY[fname] = dbg
            return result

    print("❌ No experience detected. Returning 0.0")
    EXTRACT_DEBUG_REGISTRY[fname] = dbg
    return 0.0

# ----------------------------------------------------------------------------------------
# MANDATORY-FIRST SCORING helper and UI code (kept minimal for clarity)
# ----------------------------------------------------------------------------------------
def compute_score_from_sets(
    resume_skillset: set,
    resume_exp: float,
    jd_all_skills: set,
    jd_mandatory: set,
    jd_optional: set,
    jd_min_exp: int,
):
    matched_mandatory = sorted(list(jd_mandatory & resume_skillset))
    missing_mandatory = sorted(list(jd_mandatory - resume_skillset))
    matched_optional = sorted(list(jd_optional & resume_skillset))
    matched_all = sorted(list(jd_all_skills & resume_skillset))

    ratio = min(resume_exp / jd_min_exp, 1.0) if jd_min_exp > 0 else 1.0

    if jd_mandatory:
        if len(missing_mandatory) > 0:
            score = int(20 * ratio)
            return {
                "score": score,
                "matched_mandatory": matched_mandatory,
                "missing_mandatory": missing_mandatory,
                "matched_optional": matched_optional,
                "matched_all": matched_all,
                "exp_years": round(resume_exp, 1),
                "exp_met": resume_exp >= jd_min_exp if jd_min_exp > 0 else True,
                "status": "rejected",
                "reason": "Mandatory skills missing",
                "resume_skillset": sorted(list(resume_skillset)),
            }

        mand_cov = len(matched_mandatory) / max(1, len(jd_mandatory))
        opt_cov = len(matched_optional) / max(1, len(jd_optional))
        score = round(50 * mand_cov + 30 * opt_cov + 20 * ratio)
        status = "selected" if (resume_exp >= jd_min_exp if jd_min_exp > 0 else True) and score >= 50 else "rejected"
        reason = "OK" if status == "selected" else ("Experience gap" if resume_exp < jd_min_exp else "Low score")
    else:
        cov = len(matched_all) / max(1, len(jd_all_skills))
        score = round(70 * cov + 30 * ratio)
        status = "selected" if (resume_exp >= jd_min_exp if jd_min_exp > 0 else True) and score >= 50 else "rejected"
        reason = "OK" if status == "selected" else ("Experience gap" if resume_exp < jd_min_exp else "Low score")

    if resume_exp == 0.0:
        status = "needs review"
        reason = "Experience not found (parser)"

    return {
        "score": score,
        "matched_mandatory": matched_mandatory,
        "missing_mandatory": missing_mandatory,
        "matched_optional": matched_optional,
        "matched_all": matched_all,
        "exp_years": round(resume_exp, 1),
        "exp_met": resume_exp >= jd_min_exp if jd_min_exp > 0 else True,
        "status": status,
        "reason": reason,
        "resume_skillset": sorted(list(resume_skillset)),
    }

def format_exp_years(exp_float):
    years = int(exp_float)
    months = int(round((exp_float - years) * 12))
    if months == 0:
        return f"{years} years"
    return f"{years} years {months} month{'s' if months > 1 else ''}"

# ----------------------------------------------------------------------------------------
# STREAMLIT UI (kept minimal; unchanged style from previous iteration)
# ----------------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="HIRE HUB — Robust")
APP_BG_COLOR = "#f4f8ff"

st.markdown("""
<style>
.use-jd-btn-wrap div.stButton > button:first-child {
    background-color: #16a34a !important;  /* Green */
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
    transition: all 0.3s ease-in-out !important;
}
.use-jd-btn-wrap div.stButton > button:first-child:hover {
    background-color: #22c55e !important;  /* Lighter green on hover */
    transform: scale(1.05);
    box-shadow: 0 6px 14px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
.stApp {{ background: linear-gradient(180deg, {APP_BG_COLOR} 0%, #ffffff 100%); }}
.header {{ background: #007acc; padding: 18px; border-radius: 12px; color: white; }}
.card {{ background: white; border-radius: 12px; padding: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); margin-bottom:12px; }}
.hirehub-table {{ border-collapse: collapse; width: 100%; table-layout: auto; font-family: Arial, sans-serif; font-size: 13px; }}
.hirehub-table th, .hirehub-table td {{ border: 1px solid #e2e8f0; padding: 8px; vertical-align: top; text-align: left; white-space: normal; }}
.hirehub-table th {{ background: #f1f5f9; font-weight: 600; }}
.hirehub-wrapper {{ overflow-x: auto; width: 100%; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; color: white; }}
.badge.green {{ background-color: #16a34a; }} /* Selected */
.badge.orange {{ background-color: #f59e0b; }} /* Needs Review — Manual */
.badge.red {{ background-color: #dc2626; }} /* Rejected */
.small-note {{ color:#64748b; font-size:12px; }}
.jd-inline-card {{
  max-width: 600px;
  margin: 10px auto 18px auto;
  background: #ffffff;
  border-radius: 18px;
  padding: 20px 18px 16px 18px;
  box-shadow: 0 12px 40px rgba(15,23,42,0.12);
}}
.header {{
  background: linear-gradient(135deg, #0ea5e9 0%, #1f6feb 60%, #9333ea 100%);
  padding: 18px 22px;
  border-radius: 12px;
  color: #fff;
  box-shadow: var(--shadow);
  position: sticky;
  top: 8px;
  z-index: 10;
}}
.jd-modal-title {{
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 4px;
}}
.jd-modal-sub {{
  font-size: 13px;
  color: #6b7280;
  margin-bottom: 12px;
}}
.jd-modal-footer-text {{
  font-size: 11px;
  color: #9ca3af;
  margin-top: 4px;
}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""<div class="header"><h2 style="margin:0; font-weight:700;">📄 HIRE HUB — Resume Shortlister</h2></div>""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Uploads")

# Reset & JD form state
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "jd_form_payload" not in st.session_state:
    st.session_state.jd_form_payload = None
if "jd_mandatory_from_file" not in st.session_state:
    st.session_state.jd_mandatory_from_file = set()
if "show_jd_modal" not in st.session_state:
    st.session_state.show_jd_modal = False
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "results_html_table" not in st.session_state:
    st.session_state.results_html_table = None
if "show_results_now" not in st.session_state:
    st.session_state.show_results_now = False
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None

if st.sidebar.button("🔄 Reset All Uploads"):
    for key in ["jd_file", "resume_files", "resume_zip"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.uploader_key += 1
    st.session_state.jd_form_payload = None
    st.session_state.jd_mandatory_from_file = set()
    st.session_state.results_df = None
    st.session_state.results_html_table = None
    st.session_state.show_results_now = False
    st.session_state.show_jd_modal = False
    st.rerun()

st.sidebar.markdown("Upload JD & multiple resumes (or a zip containing resumes).")
jd_file = st.sidebar.file_uploader(
    "📘 Upload Job Description (PDF/TXT)",
    type=['pdf', 'txt'],
    key=f"jd_{st.session_state.uploader_key}"
)
if jd_file:
    st.session_state.jd_file = jd_file

resume_files = st.sidebar.file_uploader(
    "🧾 Upload Resumes (PDF/DOCX/TXT) — select multiple",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True,
    key=f"resumes_{st.session_state.uploader_key}"
)
if resume_files:
    st.session_state.resume_files = resume_files

resume_zip = st.sidebar.file_uploader(
    "📁 (Optional) Upload Resumes Folder (.zip)",
    type=['zip'],
    key=f"zip_{st.session_state.uploader_key}"
)
if resume_zip:
    st.session_state.resume_zip = resume_zip

aggressive_edu_exclusion = st.sidebar.checkbox("⚡ Aggressive Education Date Exclusion", value=True)
process_button = st.sidebar.button("🚀 Start Shortlisting", type="primary")

# Center JD Form button
st.markdown('<div style="text-align:center; margin: 18px 0 4px 0;">', unsafe_allow_html=True)
open_modal = st.button("✅ Use JD Form", key="open_jd_modal", type="secondary", help="Fill JD details without a file")
st.markdown('</div>', unsafe_allow_html=True)
if open_modal:
    st.session_state.show_jd_modal = True
    st.rerun()

# JD inline form
if st.session_state.show_jd_modal:
    st.markdown('<div class="jd-modal-title">Job Description Form</div>', unsafe_allow_html=True)
    st.markdown('<div class="jd-modal-sub">Fill in JD details if you don\'t have a JD file, or want to override it.</div>', unsafe_allow_html=True)
    with st.form("jd_form_modal"):
        role_title = st.text_input("Role Title", placeholder="e.g., Embedded Linux Engineer")
        min_exp_years = st.number_input("Minimum Experience (years)", min_value=0, max_value=50, value=0, step=1)
        jd_mandatory_str = st.text_area("Mandatory Skills (comma-separated)", placeholder="e.g., Embedded C, Linux, Device Tree, U-Boot")
        jd_optional_str = st.text_area("Optional/Nice-to-have Skills (comma-separated)", placeholder="e.g., Yocto, SPI, I2C, UART")
        col_apply, col_cancel = st.columns([1,1])
        use_form = col_apply.form_submit_button("Apply JD Form", type="primary")
        cancel_form = col_cancel.form_submit_button("Cancel")

        if use_form:
            def canonize_list(s):
                items = [re.sub(r'\s+', ' ', x.strip().lower()) for x in s.split(',') if x.strip()]
                canonical = set()
                text = " " + " , ".join(items) + " "
                canonical = get_keywords_from_jd(text)
                for x in items:
                    if x not in canonical:
                        canonical.add(x)
                return canonical

            st.session_state.jd_form_payload = {
                "role": role_title.strip(),
                "min_exp": int(min_exp_years),
                "mandatory": canonize_list(jd_mandatory_str),
                "optional": canonize_list(jd_optional_str),
            }

            # Build canonical form_text for hashing and cache the form-derived keywords
            form_text = json.dumps(st.session_state.jd_form_payload, sort_keys=True)
            form_hash = _text_hash(form_text)
            st.session_state.jd_hash = form_hash
            jd_keywords_from_form = set(st.session_state.jd_form_payload["mandatory"]) | set(st.session_state.jd_form_payload["optional"])
            st.session_state.jd_keywords_cached = set(jd_keywords_from_form)

            st.session_state.show_jd_modal = False
            st.toast("JD Form applied and cached.", icon="✅")
            st.rerun()

        if cancel_form:
            st.session_state.show_jd_modal = False
            st.rerun()

    st.markdown('<div class="jd-modal-footer-text">Tip: Once applied, this JD form will override any uploaded JD file unless you reset uploads.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📘 Job Description", "🧾 Resumes"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Step 1 — Job Description")

    jd_text = ""
    jd_from_form = st.session_state.jd_form_payload is not None
    if jd_from_form:
        p = st.session_state.jd_form_payload
        st.success("Using JD **Form** (overrides file).")
        st.markdown(f"**Role**: {p['role'] or '—'}")
        st.markdown(f"**Min Experience**: **{p['min_exp']} years**")
        st.markdown(f"**Mandatory (form)**: {', '.join(sorted(list(p['mandatory']))) or '—'}")
        st.markdown(f"**Optional (form)**: {', '.join(sorted(list(p['optional']))) or '—'}")
        jd_keywords = (p["mandatory"] | p["optional"])
        jd_min_exp = p["min_exp"]
        st.info("You can still upload a JD file, but the form values from the center-button JD Form will be used.")
    else:
        if jd_file:
            if jd_file.type == "application/pdf":
                jd_text = extract_text_from_pdf(jd_file)
            else:
                jd_text = extract_text_from_txt(jd_file)
            st.text_area("JD Preview (first 2000 chars):", jd_text[:2000], height=220)

            # Provide a force-reextract button near preview
            col_a, col_b = st.columns([1, 1])
            with col_a:
                reextract = st.button("🔁 Re-extract JD Skills (force)")
            with col_b:
                st.markdown("")  # placeholder for balanced layout

            # compute hash & use cache: only call Gemini if JD changed
            new_hash = _text_hash(jd_text)
            if reextract:
                # forced re-extraction (ignore cache)
                with st.spinner("Re-extracting JD skills..."):
                    jd_keywords = get_keywords_from_jd(jd_text)
                st.success("Re-extracted JD skills (forced).")
            else:
                if new_hash != st.session_state.jd_hash:
                    # JD changed — extract and cache
                    with st.spinner("Extracting JD skills (this may take a moment)..."):
                        jd_keywords = get_keywords_from_jd(jd_text)
                    st.success("JD skills extracted and cached.")
                else:
                    # reuse cached
                    jd_keywords = set(st.session_state.jd_keywords_cached)
                    st.info("Using cached JD skills (no change detected).")

            # Try to infer min exp from text
            jd_min_exp = 0
            for pat in [r'(\d+)\s*\+\s*years', r'(\d+)\s*\-\s*\d+\s*years',
                        r'minimum\s*of\s*(\d+)\s*years', r'at\s*least\s*(\d+)\s*years',
                        r'(\d+)\s*years\s*of\s*experience']:
                found = re.findall(pat, jd_text)
                if found:
                    try:
                        jd_min_exp = int(found[0])
                        break
                    except:
                        pass

            st.markdown("#### Pick Mandatory Skills (from detected)")
            picked = st.multiselect(
                "Mark mandatory skills (optional — used as gate)",
                sorted(list(jd_keywords)),
                default=sorted(list(st.session_state.jd_mandatory_from_file or []))
            )
            st.caption("These will be checked first. If any are missing in a resume → Rejected.")
            if st.button("Apply Mandatory Selection"):
                st.session_state.jd_mandatory_from_file = set(picked)
                st.success("Saved mandatory selection.")

            st.markdown(f"**Detected JD Skills** ({len(jd_keywords)}): " + (", ".join(sorted(list(jd_keywords))) if jd_keywords else "—"))
            st.markdown(f"**Min Experience (parsed)**: **{jd_min_exp} years**")
        else:
            jd_keywords = set()
            jd_min_exp = 0
            st.info("Upload JD (PDF/TXT) or click **Use JD Form** in the center of the page.")

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Step 2 — Upload Resumes")
    if st.session_state.get("resume_files"):
        st.success(f"{len(st.session_state.resume_files)} resume(s) uploaded (individual).")
        for r in st.session_state.resume_files:
            st.markdown(f"- 📄 **{r.name}**")
    if st.session_state.get("resume_zip"):
        st.success("Zip uploaded.")
        st.markdown(f"- 📦 **{st.session_state.resume_zip.name}**")
    if not st.session_state.get("resume_files") and not st.session_state.get("resume_zip"):
        st.info("No resumes uploaded yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# Result renderer and other UI helpers (kept minimal) ---------------------------------------
def render_results_block(df, jd_min_exp_value, jd_all_skills, jd_mandatory, jd_optional, key_suffix: str = "default"):
    st.success(f"Processed {len(df)} resume(s).")
    if not df.empty:
        top_k = df.head(3)
        cols = st.columns(3)
        for c, (_, r) in zip(cols, top_k.iterrows()):
            c.markdown(f"**{r['Candidate Name']}**", unsafe_allow_html=True)
            c.markdown(f"Score: **{r['Score']}** • Exp: **{r['Years Experience']}**", unsafe_allow_html=True)
            c.markdown(r['Status'], unsafe_allow_html=True)
            c.caption((r['Matched Mandatory'][:120] + ("..." if len(r['Matched Mandatory']) > 120 else "")) or "—")

        st.markdown("### Shortlisted Candidates")

        html_table = df[[
            "Candidate Name",
            "Score",
            "Years Experience",
            "Status",
            "Matched Mandatory Count",
            "Mandatory Missing Count",
            "Matched Mandatory",
            "Mandatory Missing",
            "Matched Optional",
            "Matched (All JD Skills)",
            "Unmatched (All JD Skills)",
        ]].to_html(index=False, escape=False, classes="hirehub-table")
        wrapper = f'<div class="hirehub-wrapper">{html_table}</div>'
        st.markdown(wrapper, unsafe_allow_html=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Shortlisted')
        buf.seek(0)

        st.download_button(
            "📥 Download Shortlisted Candidates (Excel)",
            buf,
            file_name="shortlisted_candidates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_excel_{key_suffix}",
        )
        st.session_state.results_html_table = wrapper
    else:
        st.info("No candidates to display after filters.")

# Helper: read zip
def read_zip_to_files(zip_file):
    try:
        zip_file.seek(0)
        z = zipfile.ZipFile(io.BytesIO(zip_file.read()))
        files = []
        for name in z.namelist():
            if name.lower().endswith(('.pdf', '.docx', '.txt')):
                try:
                    data = z.read(name)
                    bio = io.BytesIO(data)
                    bio.name = os.path.basename(name)
                    files.append(bio)
                except:
                    continue
        return files
    except Exception as e:
        st.error(f"Failed to process zip file: {e}")
        return []

# ----------------------------------------------------------------------------------------
# PROCESS ACTION — uses Gemini-first extractors (cache-aware)
# ----------------------------------------------------------------------------------------
if process_button:
    combined_resumes = list(st.session_state.get("resume_files") or [])
    if st.session_state.get("resume_zip"):
        combined_resumes.extend(read_zip_to_files(st.session_state.resume_zip))

    jd_from_form = st.session_state.jd_form_payload is not None

    if not (jd_from_form or st.session_state.get("jd_file")):
        st.warning("Please provide a Job Description via Form or File.")
    elif not combined_resumes:
        st.warning("Please upload at least one resume (individual files or a zip).")
    else:
        # Resolve JD payload
        if jd_from_form:
            p = st.session_state.jd_form_payload
            jd_min_exp = p["min_exp"]
            jd_mandatory = set(p["mandatory"])
            jd_optional = set(p["optional"])
            jd_all_skills = jd_mandatory | jd_optional
        else:
            # Use cache if present; otherwise compute & cache
            jd_file = st.session_state.jd_file
            if jd_file.type == "application/pdf":
                jd_text = extract_text_from_pdf(jd_file)
            else:
                jd_text = extract_text_from_txt(jd_file)

            if st.session_state.get("jd_keywords_cached"):
                jd_all_skills = set(st.session_state.jd_keywords_cached)
            else:
                # worst-case: compute & cache
                with st.spinner("Extracting JD skills..."):
                    jd_all_skills = set(get_keywords_from_jd(jd_text))
                st.session_state.jd_keywords_cached = set(jd_all_skills)
                st.session_state.jd_hash = _text_hash(jd_text)

            jd_mandatory = set(st.session_state.jd_mandatory_from_file or set())
            jd_optional = jd_all_skills - jd_mandatory
            jd_min_exp = 0
            for pat in [r'(\d+)\s*\+\s*years', r'(\d+)\s*\-\s*\d+\s*years',
                        r'minimum\s*of\s*(\d+)\s*years', r'at\s*least\s*(\d+)\s*years',
                        r'(\d+)\s*years\s*of\s*experience']:
                found = re.findall(pat, jd_text)
                if found:
                    try:
                        jd_min_exp = int(found[0])
                        break
                    except:
                        pass

        # ------------------ Process resumes (with percent indicator) ------------------
        results = []
        total = len(combined_resumes)

        # progress UI: progress bar + percentage label
        progress_bar = st.progress(0)
        progress_label = st.empty()  # we'll use this to show "xx% — n/total processed"

        def _update_progress(i, total):
            if total <= 0:
                pct = 100
            else:
                pct = int(((i) / total) * 100)
                if pct > 100:
                    pct = 100
            # update progress bar (expects fraction 0..1)
            frac = min(max(i / max(1, total), 0.0), 1.0)
            progress_bar.progress(frac)
            # show percentage with processed count
            progress_label.markdown(f"**Progress:** {pct}% — {i}/{total} processed")

        # initialize display
        _update_progress(0, total)

        def status_badge(status, reason=""):
            color = {"selected": "green", "needs review": "orange", "rejected": "red"}[status]
            label = {
                "selected": "Selected",
                "rejected": "Rejected",
                "needs review": "Needs Review — Manual Check"
            }[status]
            return f'<span class="badge {color}">{label}</span>' + (f' <span class="small-note">({reason})</span>' if reason else "")

        for idx, rf in enumerate(combined_resumes, start=1):
            fname = getattr(rf, "name", f"resume_{idx}")
            txt = ""
            try:
                if fname.lower().endswith(".pdf"):
                    txt = extract_text_from_pdf(rf)
                elif fname.lower().endswith(".docx"):
                    txt = extract_text_from_docx(rf)
                else:
                    txt = extract_text_from_txt(rf)
            except Exception:
                txt = ""

            # 1) Skills extraction: GEMINI-first (no heavy local logic anymore)
            skills_list = get_skills_from_text(txt, max_terms=500)
            skills_set = set(skills_list)

            # 2) Experience extraction: ask LLM first, fallback to local extractor
            years_decimal, raw_exp = call_gemini_for_years(txt)
            if years_decimal == 0.0:
                # local fallback
                years_decimal = extract_experience_from_resume(txt, filename=fname, aggressive_edu=aggressive_edu_exclusion)

            # 3) Compute score using sets (ensures consistent response shape)
            scoring = compute_score_from_sets(
                resume_skillset=skills_set,
                resume_exp=years_decimal,
                jd_all_skills=jd_all_skills,
                jd_mandatory=jd_mandatory,
                jd_optional=jd_optional,
                jd_min_exp=jd_min_exp
            )

            # (your debug / print block left unchanged)
            src_info = EXTRACT_DEBUG_REGISTRY.get(fname, {})
            src = src_info.get("source", "n/a")
            print("\n" + "-"*78)
            print(f"📄 RESUME REPORT: {fname}")
            print("-"*78)
            print(f"• Matched Mandatory ({len(scoring['matched_mandatory'])}): {', '.join(scoring['matched_mandatory']) or '—'}")
            print(f"• Missing Mandatory ({len(scoring['missing_mandatory'])}): {', '.join(scoring['missing_mandatory']) or '—'}")
            if jd_optional:
                print(f"• Matched Optional ({len(scoring['matched_optional'])}): {', '.join(scoring['matched_optional']) or '—'}")
            print(f"• Experience : {format_exp_years(scoring['exp_years'])} (source: {src})")
            if src_info.get("structured_intervals"):
                print("• Intervals (structured):")
                for s,e in src_info["structured_intervals"]:
                    print(f" - {s} → {e}")
            if src_info.get("date_range_intervals"):
                print("• Intervals (date ranges):")
                for s,e in src_info["date_range_intervals"]:
                    print(f" - {s} → {e}")
            if src_info.get("explicit_values"):
                print(f"• Explicit numeric mentions: {src_info['explicit_values']}")
            print(f"• JD Min Exp : {jd_min_exp} years")
            print(f"• Score : {scoring['score']}")
            print(f"• Status : {scoring['status'].upper()} ({scoring['reason']})")
            print("-"*78)

            missing_all = sorted(list((jd_all_skills - set(scoring["matched_all"])))) if jd_all_skills else []

            results.append({
                "Candidate Name": os.path.splitext(fname)[0],
                "Score": scoring["score"],
                "Years Experience": format_exp_years(scoring["exp_years"]),
                "Status": status_badge(scoring["status"], scoring["reason"]),
                "Matched Mandatory Count": len(scoring["matched_mandatory"]),
                "Mandatory Missing Count": len(scoring["missing_mandatory"]),
                "Matched Mandatory": ", ".join(scoring["matched_mandatory"]),
                "Mandatory Missing": ", ".join(scoring["missing_mandatory"]),
                "Matched Optional": ", ".join(scoring["matched_optional"]),
                "Matched (All JD Skills)": ", ".join(scoring["matched_all"]),
                "Unmatched (All JD Skills)": ", ".join(missing_all),
                "Filename": fname
            })

            # update progress display (pass idx so we show processed count)
            _update_progress(idx, total)

        # finalize progress UI
        _update_progress(total, total)
        progress_label.markdown(f"**Progress:** 100% — Completed ({total}/{total})")

        df = pd.DataFrame(sorted(results, key=lambda x: (x['Score'], -x["Mandatory Missing Count"]), reverse=True))
        st.session_state.results_df = df

        if not df.empty:
            tk = df.head(3)
            st.session_state.last_summary = tk.to_dict('records')
        else:
            st.session_state.last_summary = None

        st.session_state.show_results_now = True
        st.toast("Shortlisting complete. Showing results…", icon="📊")
        st.rerun()

# Auto show results area
if st.session_state.show_results_now and st.session_state.results_df is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Results")
    render_results_block(
        st.session_state.results_df,
        0, set(), set(), set(),
        key_suffix="autoshow"
    )
    st.markdown('</div>', unsafe_allow_html=True)
