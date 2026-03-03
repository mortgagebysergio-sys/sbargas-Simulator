import re
import io
import math
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None


# ----------------------------
# Models
# ----------------------------
@dataclass
class Tradeline:
    creditor: str = ""
    account_number: str = ""
    account_type: str = ""   # revolving, installment, mortgage, collection, other
    status: str = ""         # open/closed/charge-off/collection/late/etc
    balance: float = 0.0
    credit_limit: float = 0.0
    last_reported: str = ""
    opened_date: str = ""
    remarks: str = ""
    bureau: str = ""
    raw_block: str = ""


@dataclass
class ProfileSummary:
    total_tradelines: int
    revolving_count: int
    installment_count: int
    mortgage_count: int
    collection_count: int
    chargeoff_count: int
    total_balance: float
    total_revolving_balance: float
    total_revolving_limit: float
    utilization_pct: float
    oldest_opened_years: float
    avg_age_years: float
    has_recent_lates: bool
    has_collections: bool
    has_chargeoffs: bool


# ----------------------------
# Regex helpers
# ----------------------------
DATE_RE = re.compile(r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])[/\-](\d{2}|\d{4})\b|\b(0?[1-9]|1[0-2])[/\-](\d{2}|\d{4})\b")
MONEY_RE = re.compile(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)")

# More forgiving account number patterns
ACCT_LABEL_RE = re.compile(r"(?:Acct(?:ount)?\s*(?:#|No\.?|Number)\s*[:\-]?\s*)([A-Za-z0-9Xx\*\-]{4,})", re.IGNORECASE)
ACCT_MASKED_RE = re.compile(r"\b(?:X{3,}|\*{3,}|#){0,2}[- ]?(?:X{2,}|\*{2,})[- ]?(\d{2,8})\b")
ACCT_ENDING_RE = re.compile(r"\b(?:ending\s*in|ends\s*with)\s*(\d{2,8})\b", re.IGNORECASE)

# Multiple balance labels
BAL_LABELS = [
    re.compile(r"(?:Current\s+Balance)\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.\d{2})?)", re.IGNORECASE),
    re.compile(r"(?:Balance)\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.\d{2})?)", re.IGNORECASE),
    re.compile(r"(?:Amt\s+Owed|Amount\s+Owed)\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.\d{2})?)", re.IGNORECASE),
    re.compile(r"(?:High\s+Balance)\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.\d{2})?)", re.IGNORECASE),
]

LIMIT_LABELS = [
    re.compile(r"(?:Credit\s*Limit)\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.\d{2})?)", re.IGNORECASE),
    re.compile(r"(?:Limit)\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.\d{2})?)", re.IGNORECASE),
]

LAST_REPORTED_LABELS = [
    re.compile(r"(?:Last\s*Reported)\s*[:\-]?\s*(" + DATE_RE.pattern + r")", re.IGNORECASE),
    re.compile(r"(?:Date\s*Reported)\s*[:\-]?\s*(" + DATE_RE.pattern + r")", re.IGNORECASE),
    re.compile(r"(?:Date\s*Updated|Updated)\s*[:\-]?\s*(" + DATE_RE.pattern + r")", re.IGNORECASE),
    re.compile(r"(?:Reported)\s*[:\-]?\s*(" + DATE_RE.pattern + r")", re.IGNORECASE),
    re.compile(r"(?:Status\s*Date)\s*[:\-]?\s*(" + DATE_RE.pattern + r")", re.IGNORECASE),
]

OPENED_LABELS = [
    re.compile(r"(?:Opened|Open\s*Date|Date\s*Opened)\s*[:\-]?\s*(" + DATE_RE.pattern + r")", re.IGNORECASE),
]

CREDITOR_LABELS = [
    re.compile(r"(?:Creditor|Furnisher|Subscriber|Company|Lender)\s*[:\-]\s*([A-Za-z0-9 &\.\-]{3,80})", re.IGNORECASE),
    re.compile(r"(?:Name)\s*[:\-]\s*([A-Za-z0-9 &\.\-]{3,80})", re.IGNORECASE),
]

STATUS_HINTS = {
    "collection": ["collection", "collections", "placed for collection"],
    "charge-off": ["charge off", "charged off", "charge-off", "co"],
    "late": ["late", "30 days", "60 days", "90 days", "120 days", "past due", "delinquent"],
    "mortgage": ["mortgage", "fha", "va", "conventional", "home loan", "mtg"],
    "revolving": ["revolving", "credit card", "bankcard", "visa", "mastercard", "discover", "amex"],
    "installment": ["installment", "auto", "student", "personal loan", "finance"],
}

BUREAU_WORDS = ["experian", "equifax", "transunion", "trans union"]


def safe_float(x: str) -> float:
    try:
        return float(x.replace(",", "").strip())
    except Exception:
        return 0.0


def find_first_date(s: str) -> str:
    m = DATE_RE.search(s or "")
    return m.group(0) if m else ""


def parse_date_to_dt(s: str) -> Optional[dt.date]:
    if not s:
        return None
    s = s.strip()
    # mm/yyyy
    if re.match(r"^\d{1,2}[/\-]\d{2,4}$", s):
        mm, yy = re.split(r"[/\-]", s)
        mm = int(mm)
        yy = int(yy)
        if yy < 100:
            yy += 2000 if yy < 70 else 1900
        return dt.date(yy, mm, 1)

    # mm/dd/yyyy
    m = re.match(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{2}|\d{4})$", s)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if yy < 100:
            yy += 2000 if yy < 70 else 1900
        try:
            return dt.date(yy, mm, dd)
        except Exception:
            return None
    return None


def years_between(d1: Optional[dt.date], d2: Optional[dt.date]) -> float:
    if not d1 or not d2:
        return 0.0
    return abs((d2 - d1).days) / 365.25


# ----------------------------
# PDF extraction
# ----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed. Add pdfplumber to requirements.txt")

    parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            # Using layout can help preserve columns on bureau PDFs
            t = p.extract_text(x_tolerance=2, y_tolerance=2, layout=True) or ""
            if t.strip():
                parts.append(t)
    return "\n\n".join(parts)


# ----------------------------
# Block detection (the real fix)
# ----------------------------
def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    # reduce crazy whitespace while keeping line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text


def detect_bureau(block: str) -> str:
    lb = block.lower()
    if "experian" in lb:
        return "Experian"
    if "equifax" in lb:
        return "Equifax"
    if "transunion" in lb or "trans union" in lb:
        return "TransUnion"
    return ""


def split_into_candidate_tradelines(text: str) -> List[str]:
    """
    Better than naive \n\n splitting:
    - Try to split on strong headers common in bureau exports
    - Then refine by recognizing repeated "Account" style structures
    """
    text = normalize_text(text)

    # Strong separators often present
    strong_seps = r"(?:\n\s*ACCOUNT\s+INFORMATION\s*\n)|(?:\n\s*TRADELINE\s*\n)|(?:\n\s*ACCOUNT\s+HISTORY\s*\n)"
    pieces = re.split(strong_seps, text, flags=re.IGNORECASE)

    # If that didn't work (some PDFs lack headers), fallback to multi-newline chunks
    if len(pieces) <= 2:
        pieces = re.split(r"\n{2,}", text)

    # Filter for tradeline-like blocks
    blocks = []
    for p in pieces:
        p = p.strip()
        if len(p) < 180:
            continue
        # Tradeline-ish signals
        lb = p.lower()
        signals = [
            "balance" in lb,
            "opened" in lb or "open date" in lb,
            "last reported" in lb or "date reported" in lb or "updated" in lb,
            "credit limit" in lb or "limit" in lb,
            "acct" in lb or "account" in lb,
            any(w in lb for w in STATUS_HINTS["collection"] + STATUS_HINTS["charge-off"]),
        ]
        if sum(1 for s in signals if s) >= 2:
            blocks.append(p)

    # Second pass: merge blocks that got split too aggressively (tiny follow-ups)
    merged = []
    buf = ""
    for b in blocks:
        if not buf:
            buf = b
            continue
        # If the new block looks like a continuation (no creditor-ish header, but has fields)
        headerish = re.search(r"^(?:[A-Z0-9 &\.\-]{4,60})$", b.splitlines()[0].strip()) is not None
        if len(b) < 300 and not headerish and ("balance" in b.lower() or "acct" in b.lower()):
            buf = buf + "\n" + b
        else:
            merged.append(buf)
            buf = b
    if buf:
        merged.append(buf)

    return merged


# ----------------------------
# Field extraction
# ----------------------------
def classify_account(block: str) -> Tuple[str, str]:
    lb = block.lower()
    status = "open"
    acct_type = "other"

    if any(h in lb for h in STATUS_HINTS["collection"]):
        return "collection", "collection"

    if any(h in lb for h in STATUS_HINTS["charge-off"]):
        status = "charge-off"

    if any(h in lb for h in STATUS_HINTS["mortgage"]):
        acct_type = "mortgage"
    elif any(h in lb for h in STATUS_HINTS["revolving"]):
        acct_type = "revolving"
    elif any(h in lb for h in STATUS_HINTS["installment"]):
        acct_type = "installment"

    if any(h in lb for h in STATUS_HINTS["late"]):
        if status == "open":
            status = "late"

    if re.search(r"\bclosed\b", lb):
        if status == "open":
            status = "closed"

    return acct_type, status


def extract_creditor(block: str) -> str:
    # 1) labeled creditor fields
    for rgx in CREDITOR_LABELS:
        m = rgx.search(block)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r"\s{2,}", " ", cand)
            if 3 <= len(cand) <= 80:
                return cand

    # 2) heuristic: first meaningful non-bureau line near top
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    top = lines[:10]

    for ln in top:
        low = ln.lower()
        if any(bw in low for bw in BUREAU_WORDS):
            continue
        # avoid pure labels
        if re.search(r"\b(balance|limit|opened|reported|account|status)\b", low):
            continue
        # take a line with letters that isn't too long
        if sum(c.isalpha() for c in ln) >= 4 and 3 <= len(ln) <= 60:
            return ln

    return "Unknown Creditor"


def extract_account_number(block: str) -> str:
    m = ACCT_LABEL_RE.search(block)
    if m:
        return m.group(1).strip()

    # masked patterns like ****1234
    m2 = ACCT_MASKED_RE.search(block)
    if m2:
        return f"****{m2.group(1).strip()}"

    m3 = ACCT_ENDING_RE.search(block)
    if m3:
        return f"****{m3.group(1).strip()}"

    # last resort: look for long token with digits + X/*
    tokens = re.findall(r"\b[A-Za-z0-9Xx\*]{6,}\b", block)
    for t in tokens[:30]:
        if sum(ch.isdigit() for ch in t) >= 3 and any(ch in t for ch in "Xx*"):
            return t
    return ""


def extract_money(block: str, patterns: List[re.Pattern]) -> float:
    for rgx in patterns:
        m = rgx.search(block)
        if m:
            return safe_float(m.group(1))

    # fallback: if "balance" exists but label parse failed, take closest money after the word
    lb = block.lower()
    idx = lb.find("balance")
    if idx != -1:
        window = block[idx: idx + 160]
        m = MONEY_RE.search(window)
        if m:
            return safe_float(m.group(1))

    return 0.0


def extract_date(block: str, patterns: List[re.Pattern]) -> str:
    for rgx in patterns:
        m = rgx.search(block)
        if m:
            return find_first_date(m.group(0))
    # fallback: use first date near "reported" or "updated"
    for key in ["last reported", "date reported", "updated", "reported", "status date"]:
        idx = block.lower().find(key)
        if idx != -1:
            window = block[idx: idx + 180]
            d = find_first_date(window)
            if d:
                return d
    return ""


def extract_remarks(block: str) -> str:
    rm = re.search(r"(?:Remarks|Comment|Comments|Narrative)\s*[:\-]\s*(.{0,160})", block, re.IGNORECASE)
    if rm:
        return rm.group(1).strip()
    # small fallback: if charge-off/collection, summarize
    lb = block.lower()
    if "charge" in lb and "off" in lb:
        return "Charge-off indicated"
    if "collection" in lb:
        return "Collection indicated"
    return ""


def parse_tradelines(text: str) -> List[Tradeline]:
    blocks = split_into_candidate_tradelines(text)
    tradelines: List[Tradeline] = []

    for b in blocks:
        creditor = extract_creditor(b)
        acct_num = extract_account_number(b)
        acct_type, status = classify_account(b)
        balance = extract_money(b, BAL_LABELS)
        limit_ = extract_money(b, LIMIT_LABELS)
        last_rep = extract_date(b, LAST_REPORTED_LABELS)
        opened = extract_date(b, OPENED_LABELS)
        remarks = extract_remarks(b)
        bureau = detect_bureau(b)

        # confidence filter: require either acct#, or a balance/limit, plus a non-unknown creditor
        strong = bool(acct_num) or (balance > 0) or (limit_ > 0)
        if creditor != "Unknown Creditor" and strong:
            tradelines.append(
                Tradeline(
                    creditor=creditor,
                    account_number=acct_num,
                    account_type=acct_type,
                    status=status,
                    balance=balance,
                    credit_limit=limit_,
                    last_reported=last_rep,
                    opened_date=opened,
                    remarks=remarks,
                    bureau=bureau,
                    raw_block=b[:2000],
                )
            )

    # de-dupe
    uniq = []
    seen = set()
    for t in tradelines:
        key = (t.creditor.lower().strip(), t.account_number.strip(), round(t.balance, 2), t.status)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)

    return uniq


# ----------------------------
# Profile summary
# ----------------------------
def summarize_profile(tradelines: List[Tradeline]) -> ProfileSummary:
    today = dt.date.today()

    revolving = [t for t in tradelines if t.account_type == "revolving"]
    installment = [t for t in tradelines if t.account_type == "installment"]
    mortgage = [t for t in tradelines if t.account_type == "mortgage"]
    collections = [t for t in tradelines if t.account_type == "collection" or "collection" in t.status.lower()]
    chargeoffs = [t for t in tradelines if "charge-off" in t.status.lower()]

    total_balance = sum(t.balance for t in tradelines)
    rev_balance = sum(t.balance for t in revolving)
    rev_limit = sum(t.credit_limit for t in revolving if t.credit_limit > 0)

    util = 0.0
    if rev_limit > 0:
        util = (rev_balance / rev_limit) * 100.0

    opened_dates = []
    for t in tradelines:
        d = parse_date_to_dt(t.opened_date)
        if d:
            opened_dates.append(d)

    oldest_years = years_between(min(opened_dates), today) if opened_dates else 0.0
    avg_age = (sum(years_between(d, today) for d in opened_dates) / len(opened_dates)) if opened_dates else 0.0

    has_recent_lates = any(t.status == "late" for t in tradelines)

    return ProfileSummary(
        total_tradelines=len(tradelines),
        revolving_count=len(revolving),
        installment_count=len(installment),
        mortgage_count=len(mortgage),
        collection_count=len(collections),
        chargeoff_count=len(chargeoffs),
        total_balance=total_balance,
        total_revolving_balance=rev_balance,
        total_revolving_limit=rev_limit,
        utilization_pct=util,
        oldest_opened_years=oldest_years,
        avg_age_years=avg_age,
        has_recent_lates=has_recent_lates,
        has_collections=len(collections) > 0,
        has_chargeoffs=len(chargeoffs) > 0,
    )


# ----------------------------
# Score engine (range-based, plus per-account deltas)
# ----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def estimate_base_score_range(summary: ProfileSummary) -> Tuple[int, int]:
    center = 680.0

    util = summary.utilization_pct
    if util <= 9:
        center += 25
    elif util <= 29:
        center += 10
    elif util <= 49:
        center -= 10
    elif util <= 74:
        center -= 35
    else:
        center -= 60

    if summary.has_collections:
        center -= 55
    if summary.has_chargeoffs:
        center -= 65
    if summary.has_recent_lates:
        center -= 55

    if summary.total_tradelines >= 8:
        center += 10
    elif summary.total_tradelines <= 2:
        center -= 20

    if summary.installment_count + summary.mortgage_count >= 1 and summary.revolving_count >= 1:
        center += 8

    if summary.oldest_opened_years >= 10:
        center += 10
    elif summary.oldest_opened_years >= 5:
        center += 5
    elif summary.oldest_opened_years <= 1:
        center -= 15

    center = clamp(center, 350, 850)

    width = 35
    if summary.total_tradelines <= 3:
        width += 15
    if summary.has_collections or summary.has_chargeoffs:
        width += 10

    lo = int(clamp(center - width, 350, 850))
    hi = int(clamp(center + width, 350, 850))
    return lo, hi


def util_points(u: float) -> float:
    if u <= 9: return 35
    if u <= 29: return 18
    if u <= 49: return 0
    if u <= 74: return -22
    return -45


def simulate_profile_changes(summary: ProfileSummary, actions: Dict[str, float]) -> Dict[str, Tuple[int, int]]:
    base_lo, base_hi = estimate_base_score_range(summary)
    base_center = (base_lo + base_hi) / 2.0

    delta_center = 0.0
    delta_uncertainty = 0.0

    target_util = clamp(actions.get("utilization_target_pct", summary.utilization_pct), 0, 120)
    delta_center += (util_points(target_util) - util_points(summary.utilization_pct))
    delta_uncertainty += 6

    if actions.get("remove_collections", 0) >= 1 and summary.has_collections:
        delta_center += 35
        delta_uncertainty += 18

    if actions.get("remove_chargeoffs", 0) >= 1 and summary.has_chargeoffs:
        delta_center += 40
        delta_uncertainty += 20

    if actions.get("remove_recent_lates", 0) >= 1 and summary.has_recent_lates:
        delta_center += 30
        delta_uncertainty += 18

    if actions.get("add_authorized_user", 0) >= 1:
        delta_center += 12
        delta_uncertainty += 10

    if actions.get("add_new_revolving", 0) >= 1:
        delta_center -= 8
        delta_uncertainty += 8

    age_months = clamp(actions.get("age_forward_months", 0), 0, 60)
    delta_center += (age_months / 12.0) * 3.0
    delta_uncertainty += 4

    new_center = clamp(base_center + delta_center, 350, 850)
    base_width = (base_hi - base_lo) / 2.0
    new_width = clamp(base_width + delta_uncertainty, 20, 120)

    new_lo = int(clamp(new_center - new_width, 350, 850))
    new_hi = int(clamp(new_center + new_width, 350, 850))

    return {
        "base_range": (base_lo, base_hi),
        "new_range": (new_lo, new_hi),
        "delta_range": (new_lo - base_lo, new_hi - base_hi),
    }


def recommend_for_tradeline(t: Tradeline) -> List[Dict]:
    """
    Per-account actions + estimated impact ranges (directional).
    """
    recs = []
    status = (t.status or "").lower()
    acct_type = (t.account_type or "").lower()
    bal = t.balance or 0.0

    if "collection" in status or acct_type == "collection":
        recs.append({
            "action": "Collection: negotiate delete / removal (PFD) OR settle & request deletion where possible",
            "why": "Collections are major derogatories; removal can move the score meaningfully.",
            "impact_lo": +15,
            "impact_hi": +55,
        })
        return recs

    if "charge-off" in status:
        recs.append({
            "action": "Charge-off: dispute inaccuracies / request update; target deletion or status improvement",
            "why": "Charge-offs are major derogatories; removal/update can improve score.",
            "impact_lo": +10,
            "impact_hi": +60,
        })

    if "late" in status:
        recs.append({
            "action": "Late payment: goodwill / dispute inaccuracies / correction",
            "why": "Removing recent lates can significantly improve score.",
            "impact_lo": +10,
            "impact_hi": +45,
        })

    if acct_type == "revolving":
        recs.append({
            "action": "Revolving: reduce utilization (target <10%, ideally 1–9%)",
            "why": "Utilization is one of the biggest controllable scoring factors.",
            "impact_lo": +5,
            "impact_hi": +35,
        })

    if acct_type in ["installment", "mortgage"] and bal > 0:
        recs.append({
            "action": "Installment/Mortgage: keep current; avoid new lates (pay on time)",
            "why": "Payment history matters most; these accounts help mix/age when clean.",
            "impact_lo": 0,
            "impact_hi": +10,
        })

    if not recs:
        recs.append({
            "action": "Review: verify reporting accuracy (balance, dates, status). Dispute if incorrect.",
            "why": "Fixing incorrect reporting can improve score.",
            "impact_lo": 0,
            "impact_hi": +20,
        })

    return recs


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Credit Report Simulator", page_icon="📈", layout="wide")
st.title("📈 Credit Report Simulator (Better Parsing + Recommendations)")
st.caption("Estimator-style simulator. Keep the UI, improve the parsing + show account-level actions and estimated impact ranges.")

with st.sidebar:
    st.header("Upload Credit Report PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    st.divider()
    show_debug = st.toggle("Show parser debug blocks", value=False)
    allow_manual_edits = st.toggle("Allow manual fixes to parsed accounts", value=True)
    st.divider()
    st.write("If balances/last reported are missing, the PDF layout may differ by bureau/export. Manual fixes help immediately, and debug blocks help tune the parser.")

if not uploaded:
    st.info("Upload a credit report PDF to begin.")
    st.stop()

if pdfplumber is None:
    st.error("Missing dependency: pdfplumber. Add it to requirements.txt and redeploy.")
    st.stop()

file_bytes = uploaded.read()
with st.spinner("Extracting text from PDF..."):
    text = extract_text_from_pdf(file_bytes)

if len(text.strip()) < 200:
    st.error("Extracted text is too short. This PDF may be image-scanned or protected (no selectable text).")
    st.stop()

with st.spinner("Parsing tradelines..."):
    tradelines = parse_tradelines(text)

summary = summarize_profile(tradelines)
base_lo, base_hi = estimate_base_score_range(summary)

# Top metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Estimated Current Range", f"{base_lo}–{base_hi}")
c2.metric("Tradelines Found", summary.total_tradelines)
c3.metric("Revolving Utilization", f"{summary.utilization_pct:.1f}%")
c4.metric("Collections", str(summary.collection_count))
c5.metric("Charge-offs", str(summary.chargeoff_count))

tab_sim, tab_recs, tab_accounts, tab_debug = st.tabs(
    ["✅ What-If Simulator", "🎯 Recommendations", "📋 Parsed Accounts", "🛠️ Parser Debug"]
)

# ----------------------------
# Simulator tab (same vibe, improved)
# ----------------------------
with tab_sim:
    st.subheader("What-If Scenarios (overall)")
    left, right = st.columns([1, 1])

    with left:
        target_util = st.slider("Target overall revolving utilization (%)", 0, 120, int(clamp(summary.utilization_pct, 0, 120)), 1)
        remove_collections = st.checkbox("Collections removed/updated", value=False)
        remove_chargeoffs = st.checkbox("Charge-offs removed/updated", value=False)
        remove_recent_lates = st.checkbox("Recent lates removed/corrected", value=False)

    with right:
        add_au = st.checkbox("Added as strong Authorized User", value=False)
        add_new_rev = st.checkbox("Opened a new revolving account", value=False)
        age_months = st.slider("Simulate time passing (months)", 0, 60, 0, 1)

    actions = {
        "utilization_target_pct": float(target_util),
        "remove_collections": 1.0 if remove_collections else 0.0,
        "remove_chargeoffs": 1.0 if remove_chargeoffs else 0.0,
        "remove_recent_lates": 1.0 if remove_recent_lates else 0.0,
        "add_authorized_user": 1.0 if add_au else 0.0,
        "add_new_revolving": 1.0 if add_new_rev else 0.0,
        "age_forward_months": float(age_months),
    }

    sim = simulate_profile_changes(summary, actions)
    new_lo, new_hi = sim["new_range"]
    d_lo, d_hi = sim["delta_range"]

    st.divider()
    a, b, c = st.columns([1, 1, 2])
    a.metric("Estimated New Range", f"{new_lo}–{new_hi}")
    b.metric("Estimated Change (range)", f"{d_lo:+d} to {d_hi:+d}")
    c.write("This is a directional estimate. Real outcomes vary by bureau/model/file.")

# ----------------------------
# Recommendations tab
# ----------------------------
with tab_recs:
    st.subheader("Account-by-account recommendations + estimated impact")
    if not tradelines:
        st.warning("No tradelines parsed. Use the Debug tab to see what the parser is seeing.")
    else:
        total_lo = 0
        total_hi = 0

        for i, t in enumerate(tradelines, start=1):
            acct_label = f"{i}. {t.creditor}  |  {t.account_number or '(no acct # parsed)'}"
            meta = []
            if t.status:
                meta.append(f"Status: {t.status}")
            if t.balance:
                meta.append(f"Bal: ${t.balance:,.2f}")
            if t.last_reported:
                meta.append(f"Last reported: {t.last_reported}")
            if t.bureau:
                meta.append(f"Bureau: {t.bureau}")

            with st.expander(acct_label, expanded=False):
                st.caption(" • ".join(meta) if meta else "")

                recs = recommend_for_tradeline(t)
                for r in recs:
                    st.markdown(f"**Do:** {r['action']}")
                    st.write(r["why"])
                    st.write(f"Estimated impact: **{r['impact_lo']:+d} to {r['impact_hi']:+d}** points (directional)")

                # For rough “stacked” estimate, only stack derogatory removals + utilization once.
                # We’ll sum only the single highest action per tradeline to avoid crazy overcounting.
                best = max(recs, key=lambda x: x["impact_hi"])
                total_lo += max(0, best["impact_lo"])
                total_hi += max(0, best["impact_hi"])

        st.divider()
        st.info(f"Very rough combined upside (not additive in real scoring): **+{total_lo} to +{total_hi}** points, depending on what actually gets removed/updated and the scoring model.")

# ----------------------------
# Parsed Accounts tab (with Manual Fix)
# ----------------------------
with tab_accounts:
    st.subheader("Parsed Accounts")
    if not tradelines:
        st.warning("No tradelines were confidently parsed from this PDF.")
    else:
        rows = []
        for t in tradelines:
            rows.append({
                "Creditor": t.creditor,
                "Account #": t.account_number,
                "Type": t.account_type,
                "Status": t.status,
                "Balance": t.balance,
                "Limit": t.credit_limit,
                "Last Reported": t.last_reported,
                "Opened": t.opened_date,
                "Bureau": t.bureau,
                "Remarks": t.remarks,
            })

        if allow_manual_edits:
            st.caption("You can click into cells and correct creditor, account #, balance, last reported, etc. The simulator will use your corrected values for this session.")
            edited = st.data_editor(
                rows,
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "Balance": st.column_config.NumberColumn(format="%.2f"),
                    "Limit": st.column_config.NumberColumn(format="%.2f"),
                }
            )

            # Apply edits back to tradelines in-memory
            for idx, r in enumerate(edited):
                tradelines[idx].creditor = str(r.get("Creditor", "") or "").strip()
                tradelines[idx].account_number = str(r.get("Account #", "") or "").strip()
                tradelines[idx].account_type = str(r.get("Type", "") or "").strip()
                tradelines[idx].status = str(r.get("Status", "") or "").strip()
                tradelines[idx].balance = float(r.get("Balance") or 0.0)
                tradelines[idx].credit_limit = float(r.get("Limit") or 0.0)
                tradelines[idx].last_reported = str(r.get("Last Reported", "") or "").strip()
                tradelines[idx].opened_date = str(r.get("Opened", "") or "").strip()
                tradelines[idx].bureau = str(r.get("Bureau", "") or "").strip()
                tradelines[idx].remarks = str(r.get("Remarks", "") or "").strip()

            # Refresh summary after manual edits
            summary = summarize_profile(tradelines)
            base_lo, base_hi = estimate_base_score_range(summary)
            st.success(f"Updated estimated current range (using edits): {base_lo}–{base_hi}")

        else:
            st.dataframe(rows, use_container_width=True, hide_index=True)

# ----------------------------
# Debug tab
# ----------------------------
with tab_debug:
    st.subheader("Parser Debug")
    st.write("If fields are still wrong, this is where we tune. It shows the raw tradeline blocks the parser is using.")
    if not show_debug:
        st.info("Enable **Show parser debug blocks** in the sidebar.")
    else:
        blocks = split_into_candidate_tradelines(text)
        st.caption(f"Candidate blocks found: {len(blocks)}")
        for i, b in enumerate(blocks[:30], start=1):
            title = f"Block #{i} — {extract_creditor(b)}"
            with st.expander(title):
                st.text(b[:6000])

st.divider()
st.caption("If your PDF is scanned (image-only) this won’t parse reliably without OCR. If you see garbled/misaligned extracted text, we’ll need to add OCR support.")
