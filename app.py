#without result tab with JD form
import os
import io
import re
import docx
import pdfplumber
import zipfile
import pandas as pd
from datetime import datetime
# from word2number import w2n # REMOVED: unused and increases packaging size
import streamlit as st

# ==============================================================
# GLOBAL DEBUG REGISTRY (for clean, structured logs)
# ==============================================================
EXTRACT_DEBUG_REGISTRY = {}  # { filename: {...} }

# ----------------------------------------------------------------------------------------------------------------------
# TEXT EXTRACTION UTILITIES
# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# JD / SKILL EXTRACTION (patterns + simple cue windows)
# ----------------------------------------------------------------------------------------------------------------------
def get_keywords_from_jd(jd_text, max_terms=150):
    if not jd_text:
        return set()
    text = jd_text.lower()
    text = re.sub(r'(https?://\S+|www\.\S+)', ' ', text)
    text = text.replace('&', ' and ')
    text = re.sub(r'[\u2010-\u2015]', '-', text)
    text = re.sub(r'[/]', ' / ', text)
    text = re.sub(r'[^a-z0-9+\#\.\-/\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    STOP = {
        'a','an','the','and','or','of','for','to','in','on','with','without','as','by','at','from','into','over',
        'is','are','was','were','be','being','been','can','could','should','would','may','might','must',
        'that','this','these','those','it','its','their','our','your','you','we','they',
        'years','year','experience','exp','minimum','at','least','plus','including','etc','such','as',
        'hands','on','hands-on','proficient','proficiency','expertise','expert','knowledge','understanding',
        'ability','capability','skills','skill','good','strong','excellent','great',
        'preferred','nice','have','nice-to-have','must','required','requirement','requirements',
        'responsibilities','responsibility','role','roles','about','job','description','candidate',
        'team','teams','collaboration','communicate','communication','problem','solving','ownership',
        'drive','driven','self','motivated','fast','paced','fast-paced','environment',
        'worked','working','work','design','develop','development','implement','implementation',
        'architecture','architectures','across','using','use','used','involves','contribute','contributions',
        'review','reviews','code','coding','test','testing','debug','debugging','issue','issues','resolve','resolution'
    }
    SKILL_PATTERNS = [
        # Languages
        ("python", r"\bpython\b"),
        ("java", r"\bjava\b"),
        ("c", r"(?<!\w)c(?!\+\+\#)(?!\-)(?!dac)"),
        ("c++", r"\bc\+\+\b"),
        ("c#", r"\bc#\b"),
        ("javascript", r"\bjavascript\b|\bjs\b(?!on)"),
        ("typescript", r"\btypescript\b|\bts\b"),
        ("go", r"\bgo(lang)?\b"),
        ("ruby", r"\bruby\b"),
        ("php", r"\bphp\b"),
        ("rust", r"\brust\b"),
        ("scala", r"\bscala\b"),
        ("kotlin", r"\bkotlin\b"),
        ("swift", r"\bswift\b"),
        ("objective-c", r"\bobjective[\- ]?c\b"),
        ("r", r"(?<!\w)r(?!\w)"),
        # Frontend
        ("react", r"\breact(\.js)?\b|\breactjs\b"),
        ("angular", r"\bangular(\.js)?\b"),
        ("vue", r"\bvue(\.js)?\b"),
        ("next.js", r"\bnext(\.js)?\b|nextjs"),
        ("svelte", r"\bsvelte\b"),
        ("html", r"\bhtml\b"),
        ("css", r"\bcss\b"),
        ("sass", r"\bsass\b|\bscss\b"),
        ("bootstrap", r"\bbootstrap\b"),
        ("tailwind", r"\btailwind(\s*css)?\b"),
        ("webpack", r"\bwebpack\b"),
        # Backend / frameworks
        ("node.js", r"\bnode(\.js)?\b"),
        ("express", r"\bexpress(\.js)?\b"),
        ("django", r"\bdjango\b"),
        ("flask", r"\bflask\b"),
        ("spring", r"\bspring\b(?!\s*boot)"),
        ("spring boot", r"\bspring\s+boot\b"),
        ("laravel", r"\blaravel\b"),
        ("asp.net", r"\basp\.?net\b"),
        # Databases
        ("sql", r"\bsql\b"),
        ("postgresql", r"\bpostgres(sql)?\b"),
        ("mysql", r"\bmy\s?sql\b|\bmysql\b"),
        ("mongodb", r"\bmongo(db)?\b"),
        ("redis", r"\bredis\b"),
        ("oracle", r"\boracle\b"),
        ("mssql", r"\bmssql\b|\bsql\s*server\b"),
        ("cassandra", r"\bcassandra\b"),
        # Cloud / containers / IaC
        ("aws", r"\baws\b|\bamazon web services\b"),
        ("azure", r"\bazure\b"),
        ("gcp", r"\bgcp\b|\bgoogle cloud( platform)?\b"),
        ("docker", r"\bdocker\b"),
        ("kubernetes", r"\bkubernetes\b|\bk8s\b"),
        ("terraform", r"\bterraform\b"),
        ("ansible", r"\bansible\b"),
        # CI/CD
        ("jenkins", r"\bjenkins\b"),
        ("gitlab-ci", r"\bgitlab[\- ]?ci\b"),
        ("github actions", r"\bgithub\s+actions\b"),
        ("circleci", r"\bcircleci\b"),
        ("ci/cd", r"\bci\s*/\s*cd\b|\bcontinuous integration\b|\bcontinuous delivery\b"),
        # Data / ML / AI
        ("pandas", r"\bpandas\b"),
        ("numpy", r"\bnumpy\b"),
        ("scikit-learn", r"\bscikit[\- ]?learn\b|\bsklearn\b"),
        ("tensorflow", r"\btensorflow\b"),
        ("pytorch", r"\bpytorch\b|\btorch\b"),
        ("xgboost", r"\bxgboost\b"),
        ("lightgbm", r"\blightgbm\b"),
        ("nlp", r"\bnlp\b|\bnatural language processing\b"),
        ("opencv", r"\bopen\s*cv\b|\bopencv\b"),
        ("spacy", r"\bspacy\b"),
        ("nltk", r"\bnltk\b"),
        ("keras", r"\bkeras\b"),
        # Messaging / streaming
        ("kafka", r"\bkafka\b"),
        ("rabbitmq", r"\brabbit\s*mq\b|\brabbitmq\b"),
        ("activemq", r"\bactive\s*mq\b|\bactivemq\b"),
        # Testing
        ("pytest", r"\bpytest\b"),
        ("unittest", r"\bunit\s*test(s)?\b|\bunittest\b"),
        ("jest", r"\bjest\b"),
        ("mocha", r"\bmocha\b"),
        ("selenium", r"\bselenium\b"),
        ("playwright", r"\bplaywright\b"),
        # VCS / Collaboration
        ("git", r"\bgit\b"),
        ("github", r"\bgithub\b"),
        ("gitlab", r"\bgitlab\b"),
        ("bitbucket", r"\bbitbucket\b"),
        ("jira", r"\bjira\b"),
        ("confluence", r"\bconfluence\b"),
        # BI / Viz
        ("tableau", r"\btableau\b"),
        ("power bi", r"\bpower\s*bi\b|\bpowerbi\b"),
        ("looker", r"\blooker\b"),
        ("d3.js", r"\bd3(\.js)?\b"),
        ("matplotlib", r"\bmatplotlib\b"),
        ("seaborn", r"\bseaborn\b"),
        # Architecture / APIs
        ("microservices", r"\bmicro\s*services?\b|\bmicroservices\b"),
        ("rest api", r"\brest\s*api(s)?\b|\brest\b"),
        ("graphql", r"\bgraphql\b"),
        ("api design", r"\bapi\s*design\b"),
        ("containerization", r"\bcontaineri[sz]ation\b"),
        # OS / scripting / data eng
        ("linux", r"\blinux\b"),
        ("bash", r"\bbash\b"),
        ("shell", r"\bshell\b"),
        ("unix", r"\bunix\b"),
        ("excel", r"\bexcel\b"),
        ("etl", r"\betl\b"),
        ("data pipeline", r"\bdata\s*pipeline(s)?\b"),
        ("spark", r"\bspark\b"),
        ("hadoop", r"\bhadoop\b"),
        ("embedded", r"\bembedded\b"),
        ("embedded c", r"\bembedded\s*c\b"),
        ("embedded linux", r"\bembedded\s+linux\b"),
        # Embedded / BSP / Boot
        ("kernel", r"\bkernel\b"),
        ("linux kernel", r"\blinux\s+kernel\b"),
        ("bsp", r"\bbsp\b|\bboard support package\b"),
        ("board bring-up", r"\bboard\s*bring[\- ]?up\b|\bbring[\- ]?up\b"),
        ("bootloader", r"\bbootloader(s)?\b"),
        ("u-boot", r"\bu[\- ]?boot\b"),
        ("device tree", r"\bdevice\s*tree(s)?\b|\bdevice[\- ]?tree(s)?\b"),
        ("init scripts", r"\binit\s*scripts?\b|\bsysvinit\b|\bsystemd\b"),
        ("rootfs", r"\brootfs\b"),
        ("busybox", r"\bbusybox\b"),
        ("yocto", r"\byocto(\s*project)?\b"),
        ("openwrt", r"\bopen\s*wrt\b|\bopenwrt\b"),
        ("buildroot", r"\bbuildroot\b"),
        ("bitbake", r"\bbitbake\b"),
        ("kernel drivers", r"\bkernel\s+drivers?\b|\bdevice\s*drivers?\b|\bdriver\s+development\b"),
        # Buses / Peripherals
        ("spi", r"\bspi\b"),
        ("i2c", r"\bi2c\b"),
        ("uart", r"\buart\b"),
        ("i2s", r"\bi2s\b"),
        ("gpio", r"\bgpio\b"),
        ("pcie", r"\bpcie\b|\bpcie\s*gen[1-6]\b"),
        ("usb", r"\busb\b"),
        ("ethernet", r"\bethernet\b"),
        ("can", r"\bcan(?!fd)\b"),
        ("canfd", r"\bcan[\s\-]?fd\b"),
        # Wireless / Networking
        ("wireless", r"\bwireless\b"),
        ("wifi", r"\bwifi\b|\bwi[\- ]?fi\b"),
        ("wi-fi", r"\bwi[\- ]?fi\b"),
        ("lte", r"\blte\b|\b4g\b"),
        ("router", r"\brouter(s)?\b"),
        ("wireless router", r"\bwireless\s+router(s)?\b"),
        ("router platforms", r"\brouter\s+platforms?\b"),
        ("networking", r"\bnetworking\b"),
        # Debug / Lab
        ("jtag", r"\bjtag\b"),
        ("gdb", r"\bgdb\b"),
        ("oscilloscope", r"\boscilloscope(s)?\b"),
        ("logic analyzer", r"\blogic\s+analy[sz]er(s)?\b"),
        ("trace32", r"\btrace\s*32\b|\btrace32\b"),
        ("kgdb", r"\bkgdb\b"),
        # Silicon / Vendors / Boards
        ("arm", r"\barm\b"),
        ("arm cortex", r"\barm\s+cortex\b"),
        ("nxp", r"\bnxp\b|\bi\.?mx\b"),
        ("i.mx", r"\bi\.?mx\b"),
        ("ti", r"\btexas instruments\b|\bti\b"),
        ("stm32", r"\bstm32\b"),
        ("beaglebone", r"\bbeaglebone\b"),
        ("raspberry pi", r"\bras(berry)?\s*pi\b"),
        ("qualcomm", r"\bqualcomm\b"),
        ("mediatek", r"\bmediatek\b"),
        ("broadcom", r"\bbroadcom\b"),
        ("renesas", r"\brenesas\b"),
        ("r-car", r"\br[\- ]?car\b"),
        # Security / Update / Toolchain
        ("secure boot", r"\bsecure\s+boot\b"),
        ("tpm", r"\btpm\b|\btrusted platform module\b"),
        ("firmware update", r"\bfirmware\s+update(s)?\b|\bupdate\s+mechanisms?\b"),
        ("cmake", r"\bcmake\b"),
        ("make", r"\bmake\b|\bgnumake\b"),
        ("gcc", r"\bgcc\b"),
        ("clang", r"\bclang\b"),
    ]

    def norm_token(s):
        s = s.lower().strip()
        s = s.replace('c plus plus', 'c++')
        s = s.replace('node js', 'node.js')
        s = s.replace('next js', 'next.js')
        s = s.replace('react js', 'react')
        s = s.replace('angular js', 'angular')
        s = s.replace('d3 js', 'd3.js')
        s = s.replace('powerbi', 'power bi')
        s = re.sub(r'[.\-\_]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    CANONICAL_KEYS = set(norm_token(name) for name, _ in SKILL_PATTERNS)
    CUE_RE = re.compile(
        r'(experience with|proficient in|hands[\- ]on (with|in)|expertise in|working knowledge of|knowledge of|familiar with|skills?:)',
        re.I
    )

    windows = []
    for m in CUE_RE.finditer(text):
        start = m.end()
        windows.append(text[start:start+300])
    windows.append(text)

    def tokenize(s):
        s = re.sub(r'[^a-z0-9+\#\.\-/\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s.split()

    def ngrams(tokens, n):
        for i in range(len(tokens)-n+1):
            yield " ".join(tokens[i:i+n])

    candidate_strings = set()
    for win in windows:
        toks = tokenize(win)
        toks = [t for t in toks if (t not in STOP or t in {'c','r','go'})]
        for n in (1,2,3):
            for ng in ngrams(toks, n):
                parts = ng.split()
                if all(p in STOP for p in parts):
                    continue
                candidate_strings.add(norm_token(ng))

    found = set()
    for canonical, patt in SKILL_PATTERNS:
        if re.search(patt, text, flags=re.IGNORECASE):
            found.add(canonical)

    for cand in list(candidate_strings):
        if cand in CANONICAL_KEYS:
            for canonical, _ in SKILL_PATTERNS:
                if norm_token(canonical) == cand:
                    found.add(canonical)
                    break
        cand2 = cand.replace(' ', '')
        for canonical, _ in SKILL_PATTERNS:
            ck = norm_token(canonical)
            if cand == ck or cand2 == ck.replace(' ', ''):
                found.add(canonical)

    sorted_found = sorted(found)
    if max_terms and len(sorted_found) > max_terms:
        sorted_found = sorted_found[:max_terms]
    return set(sorted_found)

# NEW: Skills from resume (same canonical patterns via JD extractor)
def get_skills_from_text(text: str) -> set:
    """Reuses get_keywords_from_jd logic to harvest skills from any text."""
    return get_keywords_from_jd(text, max_terms=500)

# ----------------------------------------------------------------------------------------------------------------------
# EXPERIENCE EXTRACTION (your existing debug-rich logic)
# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# MANDATORY-FIRST SCORING
# ----------------------------------------------------------------------------------------------------------------------
def format_exp_years(exp_float):
    years = int(exp_float)
    months = int(round((exp_float - years) * 12))
    if months == 0:
        return f"{years} years"
    return f"{years} years {months} month{'s' if months > 1 else ''}"

def score_resume_v2(
    resume_text: str,
    jd_all_skills: set,
    jd_mandatory: set,
    jd_optional: set,
    jd_min_exp: int,
    filename: str = "",
    aggressive_edu: bool = True,
):
    """
    Mandatory-first:
    - If any mandatory skill missing -> status='rejected' with reason 'Mandatory gap'.
    - Otherwise score:
      50% mandatory coverage + 30% optional coverage + 20% experience ratio
    If no mandatory defined:
      70% all-skills coverage + 30% experience ratio (back-compat)
    """
    if not resume_text:
        return {
            "score": 0,
            "matched_mandatory": [],
            "missing_mandatory": list(sorted(jd_mandatory)),
            "matched_optional": [],
            "matched_all": [],
            "exp_years": 0.0,
            "exp_met": False,
            "status": "needs review",
            "reason": "Empty or unreadable resume",
            "resume_skillset": [],
        }

    resume_text_lower = resume_text.lower()
    resume_skillset = get_skills_from_text(resume_text_lower)

    # Partition matches
    matched_mandatory = sorted(list(jd_mandatory & resume_skillset))
    missing_mandatory = sorted(list(jd_mandatory - resume_skillset))
    matched_optional = sorted(list(jd_optional & resume_skillset))
    matched_all = sorted(list((jd_all_skills & resume_skillset)))

    # Experience
    resume_exp = extract_experience_from_resume(resume_text, filename, aggressive_edu)
    ratio = min(resume_exp / jd_min_exp, 1.0) if jd_min_exp > 0 else 1.0

    # Gate on mandatory
    if jd_mandatory:
        if len(missing_mandatory) > 0:
            score = int(20 * ratio)  # give tiny credit for exp even if rejected
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

        # Compute score with mandatory weight
        mand_cov = len(matched_mandatory) / max(1, len(jd_mandatory))
        opt_cov = len(matched_optional) / max(1, len(jd_optional))
        score = round(50 * mand_cov + 30 * opt_cov + 20 * ratio)
        status = "selected" if (resume_exp >= jd_min_exp if jd_min_exp > 0 else True) and score >= 50 else "rejected"
        reason = "OK" if status == "selected" else ("Experience gap" if resume_exp < jd_min_exp else "Low score")
    else:
        # No mandatory defined → fall back to all-skills coverage + exp
        cov = len(matched_all) / max(1, len(jd_all_skills))
        score = round(70 * cov + 30 * ratio)
        status = "selected" if (resume_exp >= jd_min_exp if jd_min_exp > 0 else True) and score >= 50 else "rejected"
        reason = "OK" if status == "selected" else ("Experience gap" if resume_exp < jd_min_exp else "Low score")

    # Needs review heuristic: zero experience parsed
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

# ----------------------------------------------------------------------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------------------------------------------------------------------
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
/* Hover effect */
.use-jd-btn-wrap div.stButton > button:first-child:hover {
    background-color: #22c55e !important;  /* Lighter green on hover */
    transform: scale(1.05);
    box-shadow: 0 6px 14px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# >>> UPDATED: global CSS (overlay CSS removed, JD card will be inline)
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

/* JD card styling (centered block) */
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
# NEW: JD form visibility flag
if "show_jd_modal" not in st.session_state:
    st.session_state.show_jd_modal = False

# >>> ADDED: results persistence keys
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "results_html_table" not in st.session_state:
    st.session_state.results_html_table = None
if "show_results_now" not in st.session_state:
    st.session_state.show_results_now = False
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None  # store top cards info (optional)

if st.sidebar.button("🔄 Reset All Uploads"):
    for key in ["jd_file", "resume_files", "resume_zip"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.uploader_key += 1
    st.session_state.jd_form_payload = None
    st.session_state.jd_mandatory_from_file = set()
    # >>> also clear results
    st.session_state.results_df = None
    st.session_state.results_html_table = None
    st.session_state.show_results_now = False
    st.session_state.show_jd_modal = False
    st.rerun()

# Sidebar uploads
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

# ----------------------------------------------------------------------------------------------------------------------
# CENTER BUTTON — OPEN JD FORM "MODAL" (inline card)
# ----------------------------------------------------------------------------------------------------------------------
st.markdown(
    '<div style="text-align:center; margin: 18px 0 4px 0;">',
    unsafe_allow_html=True
)
open_modal = st.button("✅ Use JD Form", key="open_jd_modal", type="secondary", help="Fill JD details without a file")
st.markdown('</div>', unsafe_allow_html=True)

if open_modal:
    st.session_state.show_jd_modal = True
    st.rerun()

# ----------------------------------------------------------------------------------------------------------------------
# JD FORM INLINE CARD (same fields as before, no blocking overlay)
# ----------------------------------------------------------------------------------------------------------------------
if st.session_state.show_jd_modal:
    st.markdown('<div class="jd-modal-title">Job Description Form</div>', unsafe_allow_html=True)
    st.markdown('<div class="jd-modal-sub">Fill in JD details if you don\'t have a JD file, or want to override it.</div>', unsafe_allow_html=True)
    
    with st.form("jd_form_modal"):
        role_title = st.text_input("Role Title", placeholder="e.g., Embedded Linux Engineer")
        min_exp_years = st.number_input("Minimum Experience (years)", min_value=0, max_value=50, value=0, step=1)
        jd_mandatory_str = st.text_area(
            "Mandatory Skills (comma-separated)",
            placeholder="e.g., Embedded C, Linux, Device Tree, U-Boot"
        )
        jd_optional_str = st.text_area(
            "Optional/Nice-to-have Skills (comma-separated)",
            placeholder="e.g., Yocto, SPI, I2C, UART"
        )

        col_apply, col_cancel = st.columns([1,1])
        use_form = col_apply.form_submit_button("Apply JD Form", type="primary")
        cancel_form = col_cancel.form_submit_button("Cancel")

        if use_form:
            def canonize_list(s):
                items = [re.sub(r'\s+', ' ', x.strip().lower()) for x in s.split(',') if x.strip()]
                # Map via canonicalizer using the same patterns
                canonical = set()
                text = " " + " , ".join(items) + " "  # cheap trick to reuse detector
                canonical = get_skills_from_text(text)
                # Add any literal tokens not matched by patterns
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
            st.session_state.show_jd_modal = False
            st.toast("JD Form applied.", icon="✅")
            st.rerun()

        if cancel_form:
            st.session_state.show_jd_modal = False
            st.rerun()

    st.markdown(
        '<div class="jd-modal-footer-text">Tip: Once applied, this JD form will override any uploaded JD file unless you reset uploads.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ----- Tabs
tab1, tab2 = st.tabs(["📘 Job Description", "🧾 Resumes"])

# ----------------------------------------------------------------------------------------------------------------------
# TAB 1 — JD PREVIEW + Mandatory picker (for file flow)
# ----------------------------------------------------------------------------------------------------------------------
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

            # Extract skills from file
            jd_keywords = get_keywords_from_jd(jd_text)

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

            st.markdown(f"**Detected JD Skills** ({len(jd_keywords)}): " +
                        (", ".join(sorted(list(jd_keywords))) if jd_keywords else "—"))
            st.markdown(f"**Min Experience (parsed)**: **{jd_min_exp} years**")
        else:
            jd_keywords = set()
            jd_min_exp = 0
            st.info("Upload JD (PDF/TXT) or click **Use JD Form** in the center of the page.")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------
# TAB 2 — RESUME UPLOADS
# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# Helper: Result rendering (used both for auto-show and in Tab 3)
# ----------------------------------------------------------------------------------------------------------------------
def render_results_block(df, jd_min_exp_value, jd_all_skills, jd_mandatory, jd_optional, key_suffix: str = "default"):
    """Renders the results table, top 3 cards, and download button from a df already computed."""
    st.success(f"Processed {len(df)} resume(s).")
    if not df.empty:
        # Top-3 cards
        top_k = df.head(3)
        cols = st.columns(3)
        for c, (_, r) in zip(cols, top_k.iterrows()):
            c.markdown(f"**{r['Candidate Name']}**", unsafe_allow_html=True)
            c.markdown(f"Score: **{r['Score']}** • Exp: **{r['Years Experience']}**", unsafe_allow_html=True)
            c.markdown(r['Status'], unsafe_allow_html=True)
            c.caption((r['Matched Mandatory'][:120] + ("..." if len(r['Matched Mandatory']) > 120 else "")) or "—")

        st.markdown("### Shortlisted Candidates")

        # HTML table
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

        # Excel bytes
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

# ----------------------------------------------------------------------------------------------------------------------
# PROCESS ACTION
# ----------------------------------------------------------------------------------------------------------------------
if process_button:
    combined_resumes = list(st.session_state.get("resume_files") or [])
    if st.session_state.get("resume_zip"):
        try:
            st.session_state.resume_zip.seek(0)
            z = zipfile.ZipFile(io.BytesIO(st.session_state.resume_zip.read()))
            for fname in z.namelist():
                if fname.lower().endswith(('.pdf', '.docx', '.txt')):
                    try:
                        data = z.read(fname)
                        bio = io.BytesIO(data)
                        bio.name = os.path.basename(fname)
                        combined_resumes.append(bio)
                    except:
                        continue
        except Exception as e:
            st.error(f"Failed to process zip file: {e}")

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
            # File-based
            jd_file = st.session_state.jd_file
            if jd_file.type == "application/pdf":
                jd_text = extract_text_from_pdf(jd_file)
            else:
                jd_text = extract_text_from_txt(jd_file)
            jd_all_skills = get_keywords_from_jd(jd_text)
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

        # Process resumes
        results = []
        progress = st.progress(0)
        total = len(combined_resumes)

        def status_badge(status, reason=""):
            color = {"selected": "green", "needs review": "orange", "rejected": "red"}[status]
            label = {
                "selected": "Selected",
                "rejected": "Rejected",
                "needs review": "Needs Review — Manual Check"
            }[status]
            return f'<span class="badge {color}">{label}</span>' + (f' <span class="small-note">({reason})</span>' if reason else "")

        for i, rf in enumerate(combined_resumes):
            fname = getattr(rf, "name", f"resume_{i}")
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

            scr = score_resume_v2(
                txt,
                jd_all_skills=jd_all_skills,
                jd_mandatory=jd_mandatory,
                jd_optional=jd_optional,
                jd_min_exp=jd_min_exp,
                filename=fname,
                aggressive_edu=aggressive_edu_exclusion
            )

            src_info = EXTRACT_DEBUG_REGISTRY.get(fname, {})
            src = src_info.get("source", "n/a")
            print("\n" + "-"*78)
            print(f"📄 RESUME REPORT: {fname}")
            print("-"*78)
            print(f"• Matched Mandatory ({len(scr['matched_mandatory'])}): {', '.join(scr['matched_mandatory']) or '—'}")
            print(f"• Missing Mandatory ({len(scr['missing_mandatory'])}): {', '.join(scr['missing_mandatory']) or '—'}")
            if jd_optional:
                print(f"• Matched Optional ({len(scr['matched_optional'])}): {', '.join(scr['matched_optional']) or '—'}")
            print(f"• Experience : {format_exp_years(scr['exp_years'])} (source: {src})")
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
            print(f"• Score : {scr['score']}")
            print(f"• Status : {scr['status'].upper()} ({scr['reason']})")
            print("-"*78)

            missing_all = sorted(list((jd_all_skills - set(scr["matched_all"]))))

            results.append({
                "Candidate Name": os.path.splitext(fname)[0],
                "Score": scr["score"],
                "Years Experience": format_exp_years(scr["exp_years"]),
                "Status": status_badge(scr["status"], scr["reason"]),
                "Matched Mandatory Count": len(scr["matched_mandatory"]),
                "Mandatory Missing Count": len(scr["missing_mandatory"]),
                "Matched Mandatory": ", ".join(scr["matched_mandatory"]),
                "Mandatory Missing": ", ".join(scr["missing_mandatory"]),
                "Matched Optional": ", ".join(scr["matched_optional"]),
                "Matched (All JD Skills)": ", ".join(scr["matched_all"]),
                "Unmatched (All JD Skills)": ", ".join(missing_all),
                "Filename": fname
            })
            progress.progress((i + 1) / total)

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

# ----------------------------------------------------------------------------------------------------------------------
# AUTO SHOW RESULTS AREA
# ----------------------------------------------------------------------------------------------------------------------
if st.session_state.show_results_now and st.session_state.results_df is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Results")
    render_results_block(
        st.session_state.results_df,
        0, set(), set(), set(),
        key_suffix="autoshow"
    )
    st.markdown('</div>', unsafe_allow_html=True)
