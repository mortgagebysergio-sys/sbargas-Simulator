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


def mask_account_number(raw: str) -> str:
    raw = normalize_spaces(raw)
    if not raw:
        return ""
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 4:
        return f"****{digits[-4:]}"
    return raw[:12]


def extract_first_mm_yy(s: str) -> str:
    # captures MM/YY or MM/YYYY
    m = re.search(r"\b(\d{2}/\d{2,4})\b", s)
    return m.group(1) if m else ""


# =========================
# Tuned Tradeline Parser (for YOUR PDF layout)
# =========================
def extract_tradelines_from_report(text: str) -> List[Tradeline]:
    """
    Tuned to the exact mortgage merged report you uploaded.

    Strategy:
    1) Focus on TRADELINES section (stop at TRADE SUMMARY)
    2) Split into 'units' using repeated header:
       'Opened Reported Hi. Credit Credit Limit Reviewed ...'
    3) For each unit:
       - Find the first table row (line containing two MM/YY dates and at least one $ amount)
       - Extract opened_date, last_reported_date, balance (last $ amount in that row)
       - Extract creditor name:
          a) If first data line starts with text before a date, use that text
          b) If first line starts with a date, look for a later creditor line (text + date or text-only)
          c) Append continuation creditor line (like AUTO FINAN) when appropriate
       - Extract account number: first long digit token (>= 8 digits) in the unit
       - Extract chargeoff date from "CHARGE OFF 6/23" or "CHARGED OFF 6/23"
    """

    upper = text.upper()
    start = upper.find("TRADELINES")
    tradeline_text = text[start:] if start != -1 else text

    upper_t = tradeline_text.upper()
    end = upper_t.find("TRADE SUMMARY")
    if end != -1:
        tradeline_text = tradeline_text[:end]

    # Split into units by repeating header
    header_pat = r"Opened Reported Hi\. Credit Credit Limit Reviewed 30-59 60-89 90\+ Past Due Payment Balance"
    splits = re.split(header_pat, tradeline_text)

    # If split fails (format slightly different), fallback to older behavior
    if len(splits) <= 1:
        splits = re.split(r"Opened Reported Hi\. Credit", tradeline_text)

    units = []
    for chunk in splits:
        c = chunk.strip()
        if len(c) < 40:
            continue
        # Cut off bureau footer noise if it leaks in
        cut_markers = ["BIRCHWOOD CREDIT SERVICES", "Page ", "FILE #", "DATE COMPLETED", "SEND TO"]
        for mk in cut_markers:
            idx = c.upper().find(mk.upper())
            if idx != -1 and idx < 800:  # if footer begins early, trim there
                c = c[:idx].strip()
        if len(c) >= 40:
            units.append(c)

    tradelines: List[Tradeline] = []

    # Helper patterns
    money_pat = re.compile(r"\$[\d,]+")
    date_pat = re.compile(r"\b\d{2}/\d{2,4}\b")
    acct_digits_pat = re.compile(r"\b\d{8,}\b")

    def pick_table_row(lines: List[str]) -> str:
        # row usually has 2 dates + at least one money token
        for ln in lines[:8]:
            if len(date_pat.findall(ln)) >= 2 and money_pat.search(ln):
                return ln
        # fallback: any line with a money token
        for ln in lines[:10]:
            if money_pat.search(ln):
                return ln
        return ""

    def creditor_before_first_date(ln: str) -> str:
        # grab text before first date token
        m = re.search(r"^(.*?)(\b\d{2}/\d{2,4}\b)", ln)
        if not m:
            return ""
        name = normalize_spaces(m.group(1))
        # prevent grabbing empty/garbage
        if len(name) < 2:
            return ""
        return name

    def looks_like_creditor_continuation(ln: str) -> bool:
        # For cases like "AUTO FINAN 1/26 12/25" or "CENTURY BAN"
        s = normalize_spaces(ln)
        if not s or len(s) > 35:
            return False
        # Not a label line
        if any(x in s.upper() for x in ["DLA", "ECOA", "SOURCE", "BALANCE", "PAST DUE", "PAYMENT", "OPENED", "REPORTED"]):
            return False
        # Must have letters
        if not re.search(r"[A-Z]", s.upper()):
            return False
        return True

    for unit in units:
        raw_lines = [ln.strip() for ln in unit.splitlines() if ln.strip()]
        if not raw_lines:
            continue

        table_row = pick_table_row(raw_lines)
        opened_date = ""
        last_reported = ""
        balance = None

        if table_row:
            dates = date_pat.findall(table_row)
            if len(dates) >= 2:
                opened_date = dates[0]
                last_reported = dates[1]

            monies = money_pat.findall(table_row)
            if monies:
                # In your PDF, the last $ amount on the row is typically the balance
                balance = parse_money(monies[-1])

        # Find creditor name
        creditor = ""

        # Case A: first line has creditor + dates
        c = creditor_before_first_date(raw_lines[0])
        if c:
            creditor = c

            # Possible second line continuation (e.g., "AUTO FINAN 1/26 12/25" or "CENTURY BAN")
            if len(raw_lines) >= 2 and looks_like_creditor_continuation(raw_lines[1]):
                cont = creditor_before_first_date(raw_lines[1]) or normalize_spaces(raw_lines[1])
                # avoid appending if it is actually an account number line
                if not acct_digits_pat.search(cont):
                    creditor = normalize_spaces(f"{creditor} {cont}")

        # Case B: first line starts with dates (no creditor); look later
        if not creditor:
            for ln in raw_lines[:12]:
                # a later creditor line often looks like "CPS/MAIL 06/23 ... CHARGE OFF 6/23"
                c2 = creditor_before_first_date(ln)
                if c2:
                    creditor = c2
                    break
            if not creditor:
                # fallback: pick first non-label text line with letters
                for ln in raw_lines[:12]:
                    s = normalize_spaces(ln)
                    if re.search(r"[A-Z]", s.upper()) and not any(x in s.upper() for x in ["DLA", "ECOA", "SOURCE", "OPENED", "REPORTED"]):
                        creditor = s[:35]
                        break

        creditor = creditor or "Unknown"

        # Account number: first long digit token in unit
        acct_raw = ""
        for ln in raw_lines:
            m = acct_digits_pat.search(ln)
            if m:
                acct_raw = m.group(0)
                break
        acct_masked = mask_account_number(acct_raw)

        # Account type
        unit_up = " ".join(raw_lines).upper()
        acct_type = "Unknown"
        for t in ["AUTO", "INSTALLMENT", "REVOLVING", "COLLECTION", "OPEN", "OTHER"]:
            if re.search(rf"\b{t}\b", unit_up):
                acct_type = t.title()
                break

        # Past due (if present as $xxxx somewhere)
        past_due = None
        m = re.search(r"\bPAST DUE\b\s*\$?\s*([\d,]+)", unit_up)
        if m:
            past_due = parse_money(m.group(1))

        # Credit limit
        limit = None
        m = re.search(r"\bCREDIT LIMIT\b\s*\$?\s*([\d,]+)", unit_up)
        if m:
            limit = parse_money(m.group(1))

        # Charge-off date
        chargeoff_date = ""
        m = re.search(r"\bCHARGE\s*OFF\b\s*(\d{2}/\d{2,4})\b", unit_up)
        if not m:
            m = re.search(r"\bCHARGED\s*OFF\b\s*(\d{2}/\d{2,4})\b", unit_up)
        if m:
            chargeoff_date = m.group(1)

        # Last late date (if present)
        last_late_date = ""
        m = re.search(r"\bLAST\s+LATE\s+DATE\b\s*(\d{2}/\d{2,4})\b", unit_up)
        if m:
            last_late_date = m.group(1)

        # Status
        status = "Unknown"
        if "CHARGE OFF" in unit_up or "CHARGED OFF" in unit_up or "CHARGEOFF" in unit_up:
            status = "Charge Off"
        elif "COLLECTION" in unit_up:
            status = "Collection"
        elif (past_due or 0) > 0 or "PAST DUE" in unit_up or "DELINQUENT" in unit_up:
            status = "Past Due / Delinquent"
        elif "CUR WAS" in unit_up:
            status = "Current (prior delinquency: CUR WAS)"
        elif "PD WAS" in unit_up:
            status = "Current (prior delinquency: PD WAS)"
        elif "AS AGREED" in unit_up or "PAID" in unit_up:
            status = "As Agreed / Paid"

        # Remarks (key phrases)
        remarks = []
        for ph in [
            "ACCOUNT INFORMATION DISPUTED BY CONSUMER",
            "PROFIT AND LOSS WRITEOFF",
            "CHARGED OFF ACCOUNT",
            "COLLECTION ACCOUNT",
            "AUTHORIZED USER",
            "CLOSED",
        ]:
            if ph in unit_up:
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
                balance=balance,
                limit=limit,
                past_due=past_due,
                remarks="; ".join(remarks),
            )
        )

    # De-dupe
    seen = set()
    uniq = []
    for t in tradelines:
        key = (t.creditor.lower().strip(), t.account_number, t.opened_date, t.last_reported_date, t.balance, t.status)
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
    markers = ["CHARGE OFF", "COLLECTION", "DELINQUENT", "PAST DUE", "CUR WAS", "PD WAS", "WRITEOFF", "CHARGED OFF"]
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

    for t in tradelines:
        if not is_negative_tradeline(t):
            continue

        s = (t.status or "").upper()
        pd_amt = t.past_due or 0
        label = f"{t.creditor} ({t.account_number})".strip()

        if pd_amt > 0 or "PAST DUE" in s or "DELINQUENT" in s:
            recs.append(
                Recommendation(
                    action="Bring account current (pay past-due / establish repayment) and keep current",
                    target=label,
                    estimated_points_low=15,
                    estimated_points_high=45,
                    why="Active delinquency is a strong mortgage risk factor. Getting current can help once it updates.",
                    timeline_days=30,
                )
            )

        if "COLLECTION" in s:
            recs.append(
                Recommendation(
                    action="Attempt Pay-For-Delete (best); if not possible, settle and verify update to paid/$0",
                    target=label,
                    estimated_points_low=5,
                    estimated_points_high=25,
                    why="Deletion usually helps more than ‘paid’, but impact varies. Ensure reporting updates and avoid dispute remarks during underwriting.",
                    timeline_days=60,
                )
            )

        if "CHARGE OFF" in s:
            recs.append(
                Recommendation(
                    action="Negotiate delete/removal of derogatory remarks; otherwise settle and ensure balance updates to $0",
                    target=label,
                    estimated_points_low=3,
                    estimated_points_high=18,
                    why="Charge-offs can remain; best-case is deletion. Balance-to-$0 and remark cleanup can help modestly and supports underwriting.",
                    timeline_days=60,
                )
            )

        if "CUR WAS" in s or "PD WAS" in s:
            recs.append(
                Recommendation(
                    action="If accurate, no quick delete: focus on perfect payments + reduce utilization + resolve active derogatories first",
                    target=label,
                    estimated_points_low=0,
                    estimated_points_high=12,
                    why="Accurate lates are hard to remove quickly. Short-term gains usually come from utilization and resolving active derogatories.",
                    timeline_days=90,
                )
            )

    recs.sort(key=lambda x: (x.estimated_points_high, x.estimated_points_low), reverse=True)

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
    "Upload a mortgage-style credit report PDF. The app extracts tradelines (tuned to this report style), "
    "lists negative accounts with account #, dates, and balances, and generates a ranked action plan."
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
    st.write("- Conservative simulator, not official FICO.")
    st.write("- Text-based PDFs work best.")

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
        st.error("No tradelines detected. If the report template differs, we can tune further.")
        st.stop()

    negative_lines = [t for t in tradelines if is_negative_tradeline(t)]

    st.subheader("Negative accounts detected")
    st.write(f"Total tradelines parsed: **{len(tradelines)}** | Negative tradelines: **{len(negative_lines)}**")
    st.dataframe([asdict(t) for t in negative_lines], use_container_width=True)

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

    with st.expander("Debug: show all parsed tradelines"):
        st.dataframe([asdict(t) for t in tradelines], use_container_width=True)
