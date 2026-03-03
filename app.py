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
    account_type: str = "Unknown"
    balance: Optional[float] = None
    limit: Optional[float] = None
    status: str = "Unknown"
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


def extract_text_from_pdf(file) -> str:
    text_parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n".join(text_parts)


# =========================
# Tradeline Extraction (tuned to your merged report style)
# =========================
def extract_tradelines_from_birchwood_style(text: str) -> List[Tradeline]:
    """
    This parser is tuned for the merged/mortgage-style report you uploaded:
    - Finds TRADELINES section
    - Stops before TRADE SUMMARY
    - Splits accounts by creditor header lines (usually ALL CAPS)
    - Extracts basic fields: balance/limit/past_due/status/type/remarks
    """

    upper = text.upper()

    start = upper.find("TRADELINES")
    if start == -1:
        tradeline_text = text
    else:
        tradeline_text = text[start:]

    upper_t = tradeline_text.upper()
    end = upper_t.find("TRADE SUMMARY")
    if end != -1:
        tradeline_text = tradeline_text[:end]

    # Line cleanup
    raw_lines = tradeline_text.splitlines()
    lines = [ln.strip() for ln in raw_lines if ln.strip()]

    if not lines:
        return []

    def looks_like_creditor_line(ln: str) -> bool:
        # Creditors in this report are typically ALL CAPS and short-ish
        if len(ln) < 3 or len(ln) > 38:
            return False

        # Avoid section header lines
        bad_headers = [
            "TRADELINES", "TRADE SUMMARY", "DEROGATORY SUMMARY",
            "OPENED", "REPORTED", "REVIEWED", "HI. CREDIT", "HIGH CREDIT",
            "CREDIT LIMIT", "PAST DUE", "BALANCE", "PAYMENT", "ECOA",
            "SOURCE", "ACCOUNT", "MONTHS", "DLA"
        ]
        if any(b == ln.upper() for b in bad_headers):
            return False

        # Must contain letters
        letters = [c for c in ln if c.isalpha()]
        if not letters:
            return False

        # Mostly uppercase ratio
        upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))

        # Filter out a few non-creditor lines that are still uppercase
        blacklist_contains = ["EXPERIAN", "EQUIFAX", "TRANSUNION", "CREDIT", "REPORT"]
        if any(x in ln.upper() for x in blacklist_contains):
            return False

        return upper_ratio >= 0.85

    # Find the indices where each tradeline begins
    starts = []
    for i, ln in enumerate(lines):
        if looks_like_creditor_line(ln):
            starts.append(i)

    if not starts:
        return []

    # Build blocks
    blocks = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        block = "\n".join(lines[s:e]).strip()
        if len(block) >= 50:
            blocks.append(block)

    tradelines: List[Tradeline] = []

    for b in blocks:
        b_up = b.upper()
        first_line = b.splitlines()[0].strip()
        creditor = first_line

        # Type detection
        acct_type = "Unknown"
        for t in ["AUTO", "INSTALLMENT", "REVOLVING", "COLLECTION", "OPEN", "OTHER"]:
            if re.search(rf"\b{t}\b", b_up):
                acct_type = t.title()
                break

        # Balance
        bal = None
        m = re.search(r"\bBALANCE\b\s*\$?\s*([\d,]+)", b_up)
        if m:
            bal = parse_money(m.group(1))

        # Credit Limit
        lim = None
        m = re.search(r"\bCREDIT LIMIT\b\s*\$?\s*([\d,]+)", b_up)
        if m:
            lim = parse_money(m.group(1))

        # Past Due
        past_due = None
        m = re.search(r"\bPAST DUE\b\s*\$?\s*([\d,]+)", b_up)
        if m:
            past_due = parse_money(m.group(1))

        # Status inference
        status = "Unknown"
        if "CHARGE OFF" in b_up or "CHARGED OFF" in b_up or "CHARGEOFF" in b_up:
            status = "Charge Off"
        elif "COLLECTION" in b_up:
            status = "Collection"
        elif "CUR WAS" in b_up:
            status = "Current (prior delinquency: CUR WAS)"
        elif "PD WAS" in b_up:
            status = "Current (prior delinquency: PD WAS)"
        elif "PAST DUE" in b_up or (past_due or 0) > 0:
            status = "Past Due / Delinquent"
        elif "AS AGREED" in b_up or "PAID" in b_up:
            status = "As Agreed / Paid"

        # Remarks (light extraction)
        remarks = []
        notable_phrases = [
            "ACCOUNT INFORMATION DISPUTED BY CONSUMER",
            "PROFIT AND LOSS WRITEOFF",
            "CHARGED OFF ACCOUNT",
            "LAST LATE DATE",
            "COLLECTION ACCOUNT",
            "CLOSED",
            "AUTHORIZED USER"
        ]
        for ph in notable_phrases:
            if ph in b_up:
                remarks.append(ph.title())

        tradelines.append(
            Tradeline(
                creditor=creditor,
                account_type=acct_type,
                balance=bal,
                limit=lim,
                status=status,
                past_due=past_due,
                remarks="; ".join(remarks)
            )
        )

    # De-dupe
    seen = set()
    uniq = []
    for t in tradelines:
        key = (t.creditor.lower().strip(), (t.status or "").lower().strip(), t.balance, t.limit, t.past_due)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)

    return uniq


# =========================
# Negative Account Detection
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
# Conservative Scoring Logic (NOT FICO)
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

    # Global utilization actions
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
                        why="Utilization tier crossings are among the most reliable short-term score movers once balances report.",
                        timeline_days=30,
                    )
                )

    # Per-negative account actions
    for t in tradelines:
        if not is_negative_tradeline(t):
            continue

        s = (t.status or "").upper()
        pd_amt = t.past_due or 0

        # Active past due / delinquent
        if pd_amt > 0 or "PAST DUE" in s or "DELINQUENT" in s:
            recs.append(
                Recommendation(
                    action="Bring account current (pay past-due / establish repayment) and keep current",
                    target=t.creditor,
                    estimated_points_low=15,
                    estimated_points_high=45,
                    why="Active delinquency is one of the most punishing mortgage-risk signals. Updating to current can help once it reports.",
                    timeline_days=30,
                )
            )

        # Collections
        if "COLLECTION" in s:
            recs.append(
                Recommendation(
                    action="Attempt Pay-For-Delete (best); if not possible, settle and verify it updates to paid/$0",
                    target=t.creditor,
                    estimated_points_low=5,
                    estimated_points_high=25,
                    why="Deletion usually helps more than ‘paid collection’, but results vary by file thickness and bureau reporting.",
                    timeline_days=60,
                )
            )

        # Charge-offs
        if "CHARGE OFF" in s:
            recs.append(
                Recommendation(
                    action="Negotiate delete/removal of derogatory remarks; otherwise settle and ensure balance updates to $0",
                    target=t.creditor,
                    estimated_points_low=3,
                    estimated_points_high=18,
                    why="Charge-offs can remain; best-case is deletion. Balance-to-$0 and remark cleanup can modestly improve scoring and underwriting.",
                    timeline_days=60,
                )
            )

        # Prior delinquency indicators
        if "CUR WAS" in s or "PD WAS" in s:
            recs.append(
                Recommendation(
                    action="If accurate, no quick delete: focus on perfect payments + reduce utilization + resolve other derogatories first",
                    target=t.creditor,
                    estimated_points_low=0,
                    estimated_points_high=12,
                    why="Accurate lates are hard to remove. Most short-term improvement comes from utilization and resolving active derogatories.",
                    timeline_days=90,
                )
            )

    # Rank highest upside first
    recs.sort(key=lambda x: (x.estimated_points_high, x.estimated_points_low), reverse=True)

    # De-dupe exact duplicates
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
# Streamlit UI
# =========================
st.set_page_config(page_title="Mortgage Credit Simulator (Conservative)", layout="wide")
st.title("Mortgage Credit Simulator (Conservative, Non-FICO)")
st.caption(
    "Upload a mortgage-style credit report PDF. The app extracts tradelines (best-effort), "
    "shows negative accounts, ranks recommended actions by expected impact, and estimates 30/60/90-day score ranges."
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
    st.write("- Text-based PDFs work best. Scanned PDFs may require OCR (can be added later).")

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

    with st.spinner("Parsing tradelines from report…"):
        tradelines = extract_tradelines_from_birchwood_style(text)

    if not tradelines:
        st.error(
            "No tradelines were detected. This usually means the report format is different or the PDF text extraction is limited."
        )
        st.stop()

    negative_lines = [t for t in tradelines if is_negative_tradeline(t)]

    st.subheader("Detected tradelines (debug view)")
    st.write(f"Total tradelines parsed: **{len(tradelines)}**")
    st.dataframe([asdict(t) for t in tradelines], use_container_width=True)

    st.subheader("Negative accounts detected")
    st.write(f"Negative tradelines found: **{len(negative_lines)}**")
    if negative_lines:
        st.dataframe([asdict(t) for t in negative_lines], use_container_width=True)
    else:
        st.info("No negative tradelines detected by the current rules. If this is incorrect, we can tune the markers.")

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

    st.divider()
    st.subheader("Quick troubleshooting")
    st.write(
        "If you still only see 1 tradeline, it usually means the PDF text extraction is collapsing lines. "
        "Try a different export of the report (text-selectable), or we’ll add OCR support."
    )
