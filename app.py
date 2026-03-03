import re
import math
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import streamlit as st

# Optional: pdfplumber is the easiest + most consistent for credit bureau PDFs
# Add to requirements.txt: pdfplumber
try:
    import pdfplumber
except Exception:
    pdfplumber = None


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Tradeline:
    creditor: str = ""
    account_number: str = ""
    account_type: str = ""  # revolving, installment, mortgage, collection, other
    status: str = ""        # open/closed/charge-off/collection/late/etc
    balance: float = 0.0
    credit_limit: float = 0.0
    last_reported: str = ""
    opened_date: str = ""
    remarks: str = ""
    raw_block: str = ""     # debugging


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
# Utilities
# ----------------------------
MONEY_RE = re.compile(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)")
DATE_RE = re.compile(r"(\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:\d{2}|\d{4})\b|\b(?:0?[1-9]|1[0-2])[/\-](?:\d{2}|\d{4})\b)")
ACCT_RE = re.compile(r"(?:Acct(?:ount)?\s*(?:#|No\.?)\s*[:\-]?\s*)([Xx\*\-0-9]{4,})")
LIMIT_RE = re.compile(r"(?:Credit\s*Limit|Limit)\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.\d{2})?)", re.IGNORECASE)
BAL_RE = re.compile(r"(?:Balance|Bal(?:ance)?)\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.\d{2})?)", re.IGNORECASE)
LAST_REPORTED_RE = re.compile(r"(?:Last\s*Reported|Reported|Date\s*Reported)\s*[:\-]?\s*(" + DATE_RE.pattern + r")", re.IGNORECASE)
OPENED_RE = re.compile(r"(?:Opened|Open\s*Date|Date\s*Opened)\s*[:\-]?\s*(" + DATE_RE.pattern + r")", re.IGNORECASE)

STATUS_HINTS = {
    "collection": ["collection", "collections", "placed for collection"],
    "charge-off": ["charge off", "charged off", "charge-off", "co"],
    "late": ["late", "30 days", "60 days", "90 days", "120 days", "past due", "delinquent"],
    "mortgage": ["mortgage", "fha", "va", "conventional", "home loan", "mtg"],
    "revolving": ["revolving", "credit card", "bankcard", "visa", "mastercard", "discover", "amex"],
    "installment": ["installment", "auto", "student", "personal loan", "finance"],
}

def safe_float(x: str) -> float:
    try:
        return float(x.replace(",", "").strip())
    except Exception:
        return 0.0

def parse_date_to_dt(s: str) -> Optional[dt.date]:
    if not s:
        return None
    s = s.strip()
    # Normalize mm/yyyy -> mm/01/yyyy
    if re.match(r"^\d{1,2}[/\-]\d{2,4}$", s):
        parts = re.split(r"[/\-]", s)
        mm = int(parts[0])
        yy = int(parts[1])
        if yy < 100:
            yy += 2000 if yy < 70 else 1900
        return dt.date(yy, mm, 1)

    # mm/dd/yyyy or mm-dd-yyyy
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
    days = abs((d2 - d1).days)
    return days / 365.25

def chunk_text(text: str) -> List[str]:
    """
    Credit bureau PDFs often have tradelines in blocks.
    This chunker tries multiple heuristics to split them.
    """
    text = re.sub(r"\u00a0", " ", text)
    # Common separators: repeated newlines, or "ACCOUNT INFORMATION", "TRADELINE"
    # We'll split on strong boundaries and then re-join small fragments.
    raw_blocks = re.split(r"\n{2,}|(?:\bACCOUNT\s+INFORMATION\b)|(?:\bTRADELINE\b)", text, flags=re.IGNORECASE)
    blocks = []
    for b in raw_blocks:
        b = b.strip()
        if len(b) < 120:
            continue
        blocks.append(b)
    return blocks

def guess_creditor_name(block: str) -> str:
    # Often creditor appears near top in all caps or before "Acct #"
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    top = lines[:8]
    # Candidate: first long-ish line with letters
    for ln in top:
        if len(ln) >= 3 and sum(c.isalpha() for c in ln) >= 3:
            # Remove bureau labels
            cleaned = re.sub(r"^(Experian|Equifax|TransUnion)\b.*", "", ln, flags=re.IGNORECASE).strip()
            if cleaned and len(cleaned) <= 55:
                return cleaned
    # fallback: find label like "Creditor: XYZ"
    m = re.search(r"(?:Creditor|Furnisher)\s*[:\-]\s*([A-Za-z0-9 &\.\-]{3,60})", block, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return "Unknown Creditor"

def classify_account(block: str) -> Tuple[str, str]:
    lb = block.lower()
    status = "open"
    acct_type = "other"

    if any(h in lb for h in STATUS_HINTS["collection"]):
        acct_type = "collection"
        status = "collection"
        return acct_type, status

    if any(h in lb for h in STATUS_HINTS["charge-off"]):
        status = "charge-off"

    if any(h in lb for h in STATUS_HINTS["mortgage"]):
        acct_type = "mortgage"
    elif any(h in lb for h in STATUS_HINTS["revolving"]):
        acct_type = "revolving"
    elif any(h in lb for h in STATUS_HINTS["installment"]):
        acct_type = "installment"

    # Lates indicator
    if any(h in lb for h in STATUS_HINTS["late"]):
        if status == "open":
            status = "late"

    # Closed indicator
    if "closed" in lb and "closed date" in lb or re.search(r"\bclosed\b", lb):
        if status == "open":
            status = "closed"

    return acct_type, status

def extract_money_after_label(block: str, regex: re.Pattern) -> float:
    m = regex.search(block)
    if not m:
        return 0.0
    return safe_float(m.group(1))

def extract_date_after_label(block: str, regex: re.Pattern) -> str:
    m = regex.search(block)
    if not m:
        return ""
    # m.group(1) may include nested DATE_RE capture; find first date-like substring
    found = DATE_RE.search(m.group(0))
    return found.group(0) if found else ""

def extract_account_number(block: str) -> str:
    m = ACCT_RE.search(block)
    if m:
        return m.group(1).strip()
    # fallback: masked patterns
    m2 = re.search(r"(?:\b(?:XXXX|xxxx|X{3,})[-\s]*\d{2,6}\b)", block)
    return m2.group(0).strip() if m2 else ""

def parse_tradelines_from_text(text: str) -> List[Tradeline]:
    blocks = chunk_text(text)
    tradelines: List[Tradeline] = []

    for b in blocks:
        creditor = guess_creditor_name(b)
        acct_num = extract_account_number(b)
        acct_type, status = classify_account(b)

        balance = extract_money_after_label(b, BAL_RE)
        limit_ = extract_money_after_label(b, LIMIT_RE)

        # If not found, try a generic approach: first money after "Balance"
        if balance == 0.0:
            # sometimes "Current Balance"
            m = re.search(r"(?:Current\s+Balance)\s*[:\-]?\s*\$?\s*([0-9,]+(?:\.\d{2})?)", b, re.IGNORECASE)
            if m:
                balance = safe_float(m.group(1))

        last_reported = extract_date_after_label(b, LAST_REPORTED_RE)
        opened_date = extract_date_after_label(b, OPENED_RE)

        remarks = ""
        # short remarks: look for "Remarks:" or "Comment:"
        rm = re.search(r"(?:Remarks|Comment|Comments)\s*[:\-]\s*(.{0,120})", b, re.IGNORECASE)
        if rm:
            remarks = rm.group(1).strip()

        # Filter out obvious non-tradeline blocks by requiring either acct number OR balance/limit OR keywords
        if (acct_num or balance > 0 or limit_ > 0) and creditor != "Unknown Creditor":
            tradelines.append(
                Tradeline(
                    creditor=creditor,
                    account_number=acct_num,
                    account_type=acct_type,
                    status=status,
                    balance=balance,
                    credit_limit=limit_,
                    last_reported=last_reported,
                    opened_date=opened_date,
                    remarks=remarks,
                    raw_block=b[:1500],
                )
            )

    # De-duplicate by (creditor, acct_num, balance)
    uniq = []
    seen = set()
    for t in tradelines:
        key = (t.creditor.lower().strip(), t.account_number.strip(), round(t.balance, 2))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
    return uniq


# ----------------------------
# Scoring / Simulator (range-based)
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

    # “Recent lates” heuristic: if any block has 30/60/90 etc or "late"
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

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def estimate_base_score_range(summary: ProfileSummary) -> Tuple[int, int]:
    """
    Not a real FICO. This creates a plausible baseline range
    that responds to key factors similar to free simulators.
    """
    # Start around mid
    center = 680.0

    # Utilization impact
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

    # Derogatories
    if summary.has_collections:
        center -= 55
    if summary.has_chargeoffs:
        center -= 65
    if summary.has_recent_lates:
        center -= 55

    # Depth/mix
    if summary.total_tradelines >= 8:
        center += 10
    elif summary.total_tradelines <= 2:
        center -= 20

    if summary.installment_count + summary.mortgage_count >= 1 and summary.revolving_count >= 1:
        center += 8

    # Age
    if summary.oldest_opened_years >= 10:
        center += 10
    elif summary.oldest_opened_years >= 5:
        center += 5
    elif summary.oldest_opened_years <= 1:
        center -= 15

    center = clamp(center, 350, 850)

    # Range width depends on uncertainty
    width = 35
    if summary.total_tradelines <= 3:
        width += 15
    if summary.has_collections or summary.has_chargeoffs:
        width += 10

    lo = int(clamp(center - width, 350, 850))
    hi = int(clamp(center + width, 350, 850))
    return lo, hi

def simulate_actions(
    summary: ProfileSummary,
    actions: Dict[str, float],
) -> Dict[str, Tuple[int, int]]:
    """
    Returns dict with:
      - "new_range": (lo, hi)
      - "delta_range": (lo, hi)
    actions keys:
      utilization_target_pct
      remove_collections (0/1)
      remove_chargeoffs (0/1)
      remove_recent_lates (0/1)
      add_authorized_user (0/1)
      add_new_revolving (0/1)
      age_forward_months
    """
    base_lo, base_hi = estimate_base_score_range(summary)
    base_center = (base_lo + base_hi) / 2.0

    delta_center = 0.0
    delta_uncertainty = 0.0

    # Utilization change
    target_util = actions.get("utilization_target_pct", summary.utilization_pct)
    target_util = clamp(target_util, 0, 120)

    def util_points(u: float) -> float:
        if u <= 9: return 35
        if u <= 29: return 18
        if u <= 49: return 0
        if u <= 74: return -22
        return -45

    delta_center += (util_points(target_util) - util_points(summary.utilization_pct))
    delta_uncertainty += 6

    # Remove derogatories (these are big, but uncertain)
    if actions.get("remove_collections", 0) >= 1 and summary.has_collections:
        delta_center += 35
        delta_uncertainty += 18

    if actions.get("remove_chargeoffs", 0) >= 1 and summary.has_chargeoffs:
        delta_center += 40
        delta_uncertainty += 20

    if actions.get("remove_recent_lates", 0) >= 1 and summary.has_recent_lates:
        delta_center += 30
        delta_uncertainty += 18

    # Add AU: usually modest unless it adds age + perfect history + low util
    if actions.get("add_authorized_user", 0) >= 1:
        delta_center += 12
        delta_uncertainty += 10

    # Add new revolving: can dip short-term
    if actions.get("add_new_revolving", 0) >= 1:
        delta_center -= 8
        delta_uncertainty += 8

    # Age forward (time passes)
    age_months = actions.get("age_forward_months", 0)
    age_months = clamp(age_months, 0, 60)
    # small gradual improvement
    delta_center += (age_months / 12.0) * 3.0
    delta_uncertainty += 4

    new_center = clamp(base_center + delta_center, 350, 850)

    # New range width: base uncertainty + action uncertainty
    base_width = (base_hi - base_lo) / 2.0
    new_width = clamp(base_width + delta_uncertainty, 20, 120)

    new_lo = int(clamp(new_center - new_width, 350, 850))
    new_hi = int(clamp(new_center + new_width, 350, 850))

    delta_lo = new_lo - base_lo
    delta_hi = new_hi - base_hi

    return {
        "base_range": (base_lo, base_hi),
        "new_range": (new_lo, new_hi),
        "delta_range": (delta_lo, delta_hi),
    }


# ----------------------------
# PDF Reading
# ----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed. Add pdfplumber to requirements.txt")
    text_parts = []
    with pdfplumber.open(io_bytes(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n\n".join(text_parts)

class io_bytes:
    """Minimal wrapper so pdfplumber can open bytes like a file."""
    def __init__(self, b: bytes):
        import io
        self._bio = io.BytesIO(b)
    def read(self, *args, **kwargs):
        return self._bio.read(*args, **kwargs)
    def seek(self, *args, **kwargs):
        return self._bio.seek(*args, **kwargs)
    def tell(self, *args, **kwargs):
        return self._bio.tell(*args, **kwargs)
    def close(self):
        return self._bio.close()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Credit Report Simulator", page_icon="📈", layout="wide")

st.title("📈 Credit Report Simulator (PDF → Tradelines → What-If Score Ranges)")
st.caption("Estimator-style simulator inspired by public tools (Credit Karma / CreditWise / NerdWallet). Not a guaranteed FICO output.")

with st.sidebar:
    st.header("Upload Credit Report PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    st.divider()
    st.subheader("Display Options")
    show_debug = st.toggle("Show parsing debug blocks", value=False)
    st.divider()
    st.subheader("Disclaimer")
    st.write("This tool provides **directional estimates** and **ranges**. Real credit scoring models vary by bureau, version, and file contents.")

if not uploaded:
    st.info("Upload a credit report PDF to begin.")
    st.stop()

# Extract text
file_bytes = uploaded.read()
if not file_bytes:
    st.error("Empty file.")
    st.stop()

with st.spinner("Reading PDF and extracting text..."):
    try:
        if pdfplumber is None:
            st.error("Missing dependency: pdfplumber. Add it to requirements.txt and redeploy.")
            st.stop()
        text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        st.stop()

if len(text.strip()) < 200:
    st.error("Extracted text is too short. This PDF may be image-scanned or protected. Try a different bureau PDF export.")
    st.stop()

with st.spinner("Parsing tradelines..."):
    tradelines = parse_tradelines_from_text(text)

summary = summarize_profile(tradelines)
base_lo, base_hi = estimate_base_score_range(summary)

# --- Top metrics row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Estimated Current Range", f"{base_lo}–{base_hi}")
c2.metric("Tradelines Found", summary.total_tradelines)
c3.metric("Revolving Utilization", f"{summary.utilization_pct:.1f}%")
c4.metric("Collections", str(summary.collection_count))
c5.metric("Charge-offs", str(summary.chargeoff_count))

# Tabs for a friendly interface
tab1, tab2, tab3 = st.tabs(["✅ What-If Simulator", "📋 Parsed Accounts", "🛠️ Parser Debug"])

# ----------------------------
# Tab 1: Simulator
# ----------------------------
with tab1:
    st.subheader("What-If Scenarios (toggle + adjust)")
    st.write("Pick changes below to see an **estimated score range** and **range of improvement**.")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Utilization & Balances")
        target_util = st.slider(
            "Target overall revolving utilization (%)",
            min_value=0,
            max_value=120,
            value=int(clamp(summary.utilization_pct, 0, 120)),
            step=1,
            help="Lower utilization typically improves scores. Best results often under ~10%."
        )

        st.markdown("### Negative Items")
        remove_collections = st.checkbox("Collections updated/removed (pay-for-delete / deletion)", value=False)
        remove_chargeoffs = st.checkbox("Charge-offs removed/updated to non-derogatory", value=False)
        remove_recent_lates = st.checkbox("Recent lates corrected/removed", value=False)

    with right:
        st.markdown("### Account Strategy")
        add_au = st.checkbox("Added as strong Authorized User (low util, perfect history)", value=False)
        add_new_rev = st.checkbox("Opened a new revolving account (short-term dip possible)", value=False)

        st.markdown("### Time Passing")
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

    sim = simulate_actions(summary, actions)
    new_lo, new_hi = sim["new_range"]
    d_lo, d_hi = sim["delta_range"]

    st.divider()

    a, b, c = st.columns([1, 1, 2])
    a.metric("Estimated New Range", f"{new_lo}–{new_hi}")
    b.metric("Estimated Change (range)", f"{d_lo:+d} to {d_hi:+d}")
    c.write(
        "Tip: Free public simulators show **estimates**. This tool intentionally returns ranges to mirror that uncertainty."
    )

    st.markdown("### Action Summary")
    bullet_actions = []
    bullet_actions.append(f"- Utilization target: **{target_util}%** (was ~{summary.utilization_pct:.1f}%)")
    if remove_collections: bullet_actions.append("- Collections: **removed/updated**")
    if remove_chargeoffs: bullet_actions.append("- Charge-offs: **removed/updated**")
    if remove_recent_lates: bullet_actions.append("- Recent lates: **corrected/removed**")
    if add_au: bullet_actions.append("- Authorized User: **added**")
    if add_new_rev: bullet_actions.append("- New revolving account: **opened**")
    if age_months: bullet_actions.append(f"- Time: **+{age_months} months**")
    if len(bullet_actions) == 1:
        bullet_actions.append("- (No additional changes selected)")
    st.markdown("\n".join(bullet_actions))

# ----------------------------
# Tab 2: Parsed accounts
# ----------------------------
with tab2:
    st.subheader("Parsed Accounts")
    if not tradelines:
        st.warning("No tradelines were confidently parsed from this PDF.")
    else:
        # A clean table view
        rows = []
        for t in tradelines:
            rows.append({
                "Creditor": t.creditor,
                "Account #": t.account_number,
                "Type": t.account_type,
                "Status": t.status,
                "Balance": f"${t.balance:,.2f}" if t.balance else "",
                "Limit": f"${t.credit_limit:,.2f}" if t.credit_limit else "",
                "Last Reported": t.last_reported,
                "Opened": t.opened_date,
                "Remarks": t.remarks,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.caption("If creditor names / account # / last reported still look off, the PDF layout may require custom rules for that bureau format.")

# ----------------------------
# Tab 3: Parser debug
# ----------------------------
with tab3:
    st.subheader("Parser Debug")
    st.write("Use this to tune parsing rules. It shows the raw blocks the parser thinks are tradelines.")
    if not show_debug:
        st.info("Enable **Show parsing debug blocks** in the sidebar to view raw text blocks.")
    else:
        if not tradelines:
            st.warning("No tradelines captured. Debug: show first 5 blocks from the chunker.")
            blocks = chunk_text(text)[:5]
            for i, b in enumerate(blocks, start=1):
                with st.expander(f"Chunk #{i}"):
                    st.text(b[:5000])
        else:
            for i, t in enumerate(tradelines, start=1):
                label = f"{i}. {t.creditor} — {t.account_number or '(no acct #)'}"
                with st.expander(label):
                    st.json({k: v for k, v in asdict(t).items() if k != "raw_block"})
                    st.text(t.raw_block)

st.divider()
st.caption("If you want this to match a specific bureau PDF perfectly, upload 2–3 sample PDFs of the same export format and we can harden the regex rules.")
