import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

import streamlit as st
import pdfplumber


# =========================
# Data Models
# =========================
@dataclass
class Tradeline:
    creditor: str
    account_number: str = ""
    account_type: str = "Unknown"
    status: str = "Unknown"
    opened_date: str = ""
    last_reported_date: str = ""
    chargeoff_date: str = ""
    last_late_date: str = ""
    balance: Optional[float] = None
    limit: Optional[float] = None
    past_due: Optional[float] = None
    remarks: str = ""


@dataclass
class Recommendation:
    action: str
    target: str
    estimated_points_low: int
    estimated_points_high: int
    why: str
    timeline_days: int


# =========================
# Utilities
# =========================
def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def parse_money(s: str) -> Optional[float]:
    try:
        s = s.replace(",", "").replace("$", "").strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_text_from_pdf(file) -> str:
    text_parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n".join(text_parts)


def extract_first_date(text: str) -> str:
    """
    Extracts a date-like token. Handles common formats:
    01/15/2024, 01/2024, 2024-01-15
    """
    if not text:
        return ""
    t = text.strip()

    m = re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", t)
    if m:
        return m.group(1)

    m = re.search(r"\b(\d{1,2}/\d{4})\b", t)  # MM/YYYY
    if m:
        return m.group(1)

    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
    if m:
        return m.group(1)

    return ""


def mask_account_number(acct: str) -> str:
    """
    Keep last 4 digits if possible.
    """
    acct = normalize_spaces(acct)
    if not acct:
        return ""
    digits = re.sub(r"\D", "", acct)
    if len(digits) >= 4:
        return f"****{digits[-4:]}"
    return acct[:12]


# =========================
# Tradeline Extraction (tuned + expanded fields)
# =========================
def extract_tradelines_from_report(text: str) -> List[Tradeline]:
    """
    Tuned for the mortgage merged report style you uploaded:
    - Focus TRADELINES section
    - Split by creditor header (mostly uppercase)
    - Parse: creditor, acct#, opened, last reported, chargeoff/last late, balance, limit, past due, type, status
    """

    upper = text.upper()
    start = upper.find("TRADELINES")
    tradeline_text = text[start:] if start != -1 else text

    upper_t = tradeline_text.upper()
    end = upper_t.find("TRADE SUMMARY")
    if end != -1:
        tradeline_text = tradeline_text[:end]

    raw_lines = tradeline_text.splitlines()
    lines = [ln.rstrip() for ln in raw_lines if ln.strip()]
    if not lines:
        return []

    def looks_like_creditor_line(ln: str) -> bool:
        s = ln.strip()
        if len(s) < 3 or len(s) > 42:
            return False

        # Avoid non-creditor headers
        bad_exact = {
            "TRADELINES", "TRADE SUMMARY", "DEROGATORY SUMMARY",
            "OPENED", "REPORTED", "REVIEWED", "HI. CREDIT", "HIGH CREDIT",
            "CREDIT LIMIT", "PAST DUE", "BALANCE", "PAYMENT", "ECOA",
            "SOURCE", "ACCOUNT", "MONTHS", "DLA"
        }
        if s.upper() in bad_exact:
            return False

        # Must contain letters
        letters = [c for c in s if c.isalpha()]
        if not letters:
            return False

        # mostly uppercase letters
        upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
        if upper_ratio < 0.85:
            return False

        # blacklist common non-creditor uppercase lines
        if any(x in s.upper() for x in ["EXPERIAN", "EQUIFAX", "TRANSUNION", "CREDIT REPORT"]):
            return False

        return True

    # Find block starts
    starts = [i for i, ln in enumerate(lines) if looks_like_creditor_line(ln)]
    if not starts:
        return []

    blocks = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        block = "\n".join([ln.strip() for ln in lines[s:e]]).strip()
        if len(block) >= 50:
            blocks.append(block)

    tradelines: List[Tradeline] = []

    for b in blocks:
        b_norm = normalize_spaces(b)
        b_up = b_norm.upper()

        # Creditor is first line of original block (safer than regex)
        first_line = b.splitlines()[0].strip()
        creditor = normalize_spaces(first_line)

        # Account number: look for "ACCOUNT" or patterns like "ACCT" or "ACCOUNT #"
        acct = ""
        m = re.search(r"\b(ACCOUNT|ACCT)\s*(#|NUMBER|NO\.?)?\s*[:\-]?\s*([A-Z0-9\*\-]{4,30})\b", b_up)
        if m:
            acct = m.group(3)
        else:
            # Some reports show account number immediately under creditor line
            # Try second line if it contains digits/asterisks
            blines = [normalize_spaces(x) for x in b.splitlines() if x.strip()]
            if len(blines) >= 2:
                if re.search(r"[\d\*]{4,}", blines[1]):
                    acct = blines[1]

        acct_masked = mask_account_number(acct)

        # Dates: Opened / Reported / Charge-off / Last late
        opened_date = ""
        last_reported = ""
        chargeoff_date = ""
        last_late_date = ""

        # These labels vary; we try a few
        def find_date_after(label_pat: str) -> str:
            m = re.search(label_pat + r"\s*[:\-]?\s*([^\|]{0,30})", b_up)
            if not m:
                return ""
            return extract_first_date(m.group(1))

        opened_date = find_date_after(r"\bOPENED\b")
        last_reported = find_date_after(r"\b(REPORTED|LAST REPORTED)\b")
        chargeoff_date = find_date_after(r"\b(CHARGE\s*OFF|CHARGED\s*OFF)\b")
        last_late_date = find_date_after(r"\b(LAST\s+LATE\s+DATE|LAST\s+LATE)\b")

        # Money fields
        bal = None
        m = re.search(r"\bBALANCE\b\s*\$?\s*([\d,]+)", b_up)
        if m:
            bal = parse_money(m.group(1))

        lim = None
        m = re.search(r"\bCREDIT\s*LIMIT\b\s*\$?\s*([\d,]+)", b_up)
        if m:
            lim = parse_money(m.group(1))

        past_due = None
        m = re.search(r"\bPAST\s*DUE\b\s*\$?\s*([\d,]+)", b_up)
        if m:
            past_due = parse_money(m.group(1))

        # Type detection
        acct_type = "Unknown"
        for t in ["AUTO", "INSTALLMENT", "REVOLVING", "COLLECTION", "OPEN", "OTHER"]:
            if re.search(rf"\b{t}\b", b_up):
                acct_type = t.title()
                break

        # Status inference (expanded)
        status = "Unknown"
        if "CHARGE OFF" in b_up or "CHARGED OFF" in b_up or "CHARGEOFF" in b_up:
            status = "Charge Off"
        elif "COLLECTION" in b_up:
            status = "Collection"
        elif "PAST DUE" in b_up or (past_due or 0) > 0:
            status = "Past Due / Delinquent"
        elif "CUR WAS" in b_up:
            status = "Current (prior delinquency: CUR WAS)"
        elif "PD WAS" in b_up:
            status = "Current (prior delinquency: PD WAS)"
        elif "AS AGREED" in b_up or "PAID" in b_up:
            status = "As Agreed / Paid"

        # Remarks (pull key phrases)
        remarks = []
        notable_phrases = [
            "ACCOUNT INFORMATION DISPUTED BY CONSUMER",
            "PROFIT AND LOSS WRITEOFF",
            "CHARGED OFF ACCOUNT",
            "COLLECTION ACCOUNT",
            "CLOSED",
            "AUTHORIZED USER",
        ]
        for ph in notable_phrases:
            if ph in b_up:
                remarks.append(ph.title())

        tradelines.append(
            Tradeline(
                creditor=creditor,
                account_number=acct_masked,
                account_type=acct_type,
                status=status,
                opened_date=opened_date,
                last_reported_date=last_reported,
                chargeoff_date=chargeoff_date,
                last_late_date=last_late_date,
                balance=bal,
                limit=lim,
                past_due=past_due,
                remarks="; ".join(remarks),
            )
        )

    # De-dupe
    seen = set()
    uniq = []
    for t in tradelines:
        key = (
            t.creditor.lower().strip(),
            t.account_number.strip(),
            (t.status or "").lower().strip(),
            t.balance,
            t.limit,
            t.past_due,
            t.last_reported_date,
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)

    return uniq


# =========================
# Negative Detection
# =========================
def is_negative_tradeline(t: Tradeline) -> bool:
    s = (t.status or "").upper()
    r = (t.remarks or "").upper()
    pd = (t.past_due or 0) > 0

    markers = [
        "CHARGE OFF", "COLLECTION", "DELINQUENT", "DELINQUENCY", "PAST DUE",
        "CUR WAS", "PD WAS", "WRITEOFF", "WRITE OFF", "CHARGED OFF"
    ]
    return pd or any(m in s for m in markers) or any(m in r for m in markers)


# =========================
# Conservative scoring (NOT FICO)
# =========================
UTIL_TIERS = [
    (0.09, (12, 30)),
    (0.29, (8, 22)),
    (0.49, (4, 14)),
    (0.69, (0, 8)),
    (0.89, (-6, 2)),
    (1.00, (-12, -2)),
]


def utilization_points(util: float) -> Tuple[int, int]:
    util = max(0.0, util)
    for cap, pts in UTIL_TIERS:
        if util <= cap:
            return pts
    return (-12, -2)


def estimate_overall_utilization(tradelines: List[Tradeline]) -> Optional[float]:
    total_bal = 0.0
    total_lim = 0.0
    found = False
    for t in tradelines:
        if t.balance is not None and t.limit is not None and t.limit > 0:
            total_bal += t.balance
            total_lim += t.limit
            found = True
    if not found or total_lim <= 0:
        return None
    return total_bal / total_lim


def generate_recommendations(tradelines: List[Tradeline]) -> List[Recommendation]:
    recs: List[Recommendation] = []

    # Global utilization
    util = estimate_overall_utilization(tradelines)
    if util is not None:
        current_pts = utilization_points(util)
        for target in [0.69, 0.49, 0.29, 0.09]:
            if util > target:
                new_pts = utilization_points(target)
                low_gain = clamp(new_pts[0] - current_pts[1], 0, 60)
                high_gain = clamp(new_pts[1] - current_pts[0], 0, 80)
                recs.append(
                    Recommendation(
                        action=f"Pay down revolving utilization to ≤ {int(target*100)}% (tier crossing)",
                        target="Overall revolving utilization",
                        estimated_points_low=low_gain,
                        estimated_points_high=high_gain,
                        why="Utilization tier crossings are typically the fastest conservative score wins once balances report.",
                        timeline_days=30,
                    )
                )

    # Per negative tradeline
    for t in tradelines:
        if not is_negative_tradeline(t):
            continue

        s = (t.status or "").upper()
        pd_amt = t.past_due or 0

        # Past due
        if pd_amt > 0 or "PAST DUE" in s or "DELINQUENT" in s:
            recs.append(
                Recommendation(
                    action="Bring account current (pay past-due / establish repayment) and keep current",
                    target=f"{t.creditor} ({t.account_number})".strip(),
                    estimated_points_low=15,
                    estimated_points_high=45,
                    why="Active delinquency is a strong mortgage risk factor. Getting current can help once it updates.",
                    timeline_days=30,
                )
            )

        # Collection
        if "COLLECTION" in s:
            recs.append(
                Recommendation(
                    action="Attempt Pay-For-Delete (best); if not possible, settle and verify update to paid/$0",
                    target=f"{t.creditor} ({t.account_number})".strip(),
                    estimated_points_low=5,
                    estimated_points_high=25,
                    why="Deletion usually helps more than ‘paid’, but impact varies. Ensure reporting updates and avoid dispute remarks during underwriting.",
                    timeline_days=60,
                )
            )

        # Charge-off
        if "CHARGE OFF" in s:
            recs.append(
                Recommendation(
                    action="Negotiate delete/removal of derogatory remarks; otherwise settle and ensure balance updates to $0",
                    target=f"{t.creditor} ({t.account_number})".strip(),
                    estimated_points_low=3,
                    estimated_points_high=18,
                    why="Charge-offs can remain; best-case is deletion. Balance-to-$0 and remark cleanup can help modestly and supports underwriting.",
                    timeline_days=60,
                )
            )

        # Prior delinquency markers
        if "CUR WAS" in s or "PD WAS" in s:
            recs.append(
                Recommendation(
                    action="If accurate, no quick delete: focus on perfect payments + reduce utilization + resolve active derogatories first",
                    target=f"{t.creditor} ({t.account_number})".strip(),
                    estimated_points_low=0,
                    estimated_points_high=12,
                    why="Accurate lates are hard to remove quickly. Short-term gains usually come from utilization and resolving active derogatories.",
                    timeline_days=90,
                )
            )

    # Rank
    recs.sort(key=lambda x: (x.estimated_points_high, x.estimated_points_low), reverse=True)

    # De-dupe
    seen = set()
    out = []
    for r in recs:
        key = (r.action, r.target)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def project_scores(base_score: int, recs: List[Recommendation]) -> Dict[int, Tuple[int, int]]:
    horizons = [30, 60, 90]
    proj = {}
    for h in horizons:
        low = 0
        high = 0
        for r in recs:
            if r.timeline_days <= h:
                low += r.estimated_points_low
                high += r.estimated_points_high
        proj[h] = (clamp(base_score + low, 300, 850), clamp(base_score + high, 300, 850))
    return proj


# =========================
# UI
# =========================
st.set_page_config(page_title="Mortgage Credit Simulator (Conservative)", layout="wide")
st.title("Mortgage Credit Simulator (Conservative, Non-FICO)")

st.caption(
    "Upload a mortgage-style credit report PDF. The app extracts tradelines (best-effort), "
    "lists negative accounts with key fields (account #, last reported, etc.), "
    "and generates a ranked action plan + 30/60/90 projections."
)

left, right = st.columns([2, 1], gap="large")

with right:
    base_score = st.number_input(
        "Starting middle score (enter the client’s middle score)",
        min_value=300,
        max_value=850,
        value=660,
        step=1
    )
    st.divider()
    st.markdown("### Notes")
    st.write("- This is a conservative simulator, not an official FICO model.")
    st.write("- Text-based PDFs work best. Scanned PDFs may require OCR (optional).")

with left:
    pdf = st.file_uploader("Upload credit report PDF", type=["pdf"])
    if not pdf:
        st.stop()

    with st.spinner("Extracting text from PDF…"):
        text = extract_text_from_pdf(pdf)

    if not text.strip():
        st.error("No extractable text found. This PDF may be scanned/image-based.")
        st.stop()

    with st.expander("Preview extracted text (first 2,000 characters)"):
        st.code(text[:2000])

    with st.spinner("Parsing tradelines…"):
        tradelines = extract_tradelines_from_report(text)

    if not tradelines:
        st.error("No tradelines detected. If this report format differs, we can tune the parser.")
        st.stop()

    negative_lines = [t for t in tradelines if is_negative_tradeline(t)]

    st.subheader("Negative accounts detected")
    st.write(f"Total tradelines parsed: **{len(tradelines)}** | Negative tradelines: **{len(negative_lines)}**")

    if negative_lines:
        st.dataframe([asdict(t) for t in negative_lines], use_container_width=True)
    else:
        st.info("No negative tradelines detected by current rules.")

    recs = generate_recommendations(tradelines)

    st.subheader("Ranked actions (highest impact first)")
    st.dataframe([asdict(r) for r in recs], use_container_width=True)

    proj = project_scores(int(base_score), recs)
    st.subheader("30/60/90-day conservative projections")
    c1, c2, c3 = st.columns(3)
    for col, horizon in zip([c1, c2, c3], [30, 60, 90]):
        lo, hi = proj[horizon]
        with col:
            st.metric(f"{horizon}-day projection", f"{lo} – {hi}")

    st.divider()
    st.subheader("Plan of action per negative account")

    by_target = defaultdict(list)
    for r in recs:
        by_target[r.target].append(r)

    for target, items in by_target.items():
        st.markdown(f"### {target}")
        for r in items:
            st.markdown(
                f"- **{r.action}** — Estimated **+{r.estimated_points_low} to +{r.estimated_points_high}** "
                f"(~{r.timeline_days} days)\n"
                f"  - Why: {r.why}"
            )
