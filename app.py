
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta
from pathlib import Path
import glob
import os

st.set_page_config(page_title="Contacts Dashboard", layout="wide")

# -----------------------
# Helpers
# -----------------------
@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    """
    Try to load a bundled CSV. If not found (e.g., on Streamlit Cloud without data/),
    fall back to any CSV in the repo. If none exist, return an empty DataFrame.
    """
    candidates = [
        "data/admin_contacts.csv",
        "data/contacts.csv",
        "admin_contacts.csv",
        "contacts.csv",
    ]
    for c in candidates:
        if Path(c).exists():
            return pd.read_csv(c)

    # Fallback: any CSV anywhere in the repo
    for c in glob.glob("**/*.csv", recursive=True):
        try:
            return pd.read_csv(c)
        except Exception:
            continue

    # Nothing found
    return pd.DataFrame()

def coerce_datetime(s: pd.Series):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([], dtype=object))

def normalize_marketing(val):
    if pd.isna(val):
        return "Unknown"
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "yes", "y", "1", "accepted", "accept", "opt-in"):
            return "Accepted"
        if v in ("false", "no", "n", "0", "declined", "opt-out"):
            return "Declined"
        return "Unknown"
    if isinstance(val, (int, float, bool, np.bool_)):
        return "Accepted" if bool(val) else "Declined"
    return "Unknown"

def build_name(row):
    fn = str(row.get("First Name", "") or "").strip()
    ln = str(row.get("Last Name", "") or "").strip()
    n = (fn + " " + ln).strip()
    return n or "—"

# -----------------------
# Load data
# -----------------------
df = load_default().copy()

# If no CSV found in repo, prompt for upload before proceeding
if df.empty:
    st.warning("No bundled CSV found. Upload a .csv to get started.")
    up0 = st.file_uploader("Upload .csv", type=["csv"], key="bootstrap_uploader")
    if up0 is not None:
        try:
            df = pd.read_csv(up0).copy()
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()
    if df.empty:
        st.stop()

# Optional CSV Upload (replaces the bundled dataset for this session)
with st.expander("➕ Upload a different CSV (optional)", expanded=False):
    up = st.file_uploader("Upload .csv", type=["csv"], key="secondary_uploader",
                          help="Drop a CSV to analyze it instead of the bundled data.")
    if up is not None:
        try:
            df_uploaded = pd.read_csv(up)
            st.success(f"Loaded {len(df_uploaded):,} rows from uploaded file.")
            df = df_uploaded.copy()
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    if st.button("Reset to bundled dataset", use_container_width=True):
        df = load_default().copy()
        st.rerun()

# Column mapping
COL_TIME = next((c for c in df.columns if c.lower() == "time" or "date" in c.lower()), None)
COL_CREATED = next((c for c in df.columns if c.lower().strip() == "created by user"), None) or \
              next((c for c in df.columns if "created by" in c.lower()), None)
COL_ACCEPT = next((c for c in df.columns if "accept marketing" in c.lower()), None)

if COL_TIME is None or COL_CREATED is None:
    st.error("Could not infer required columns. Expected 'Time' (or any date-like column) and 'Created By User'.")
    st.stop()

# Prepare fields
df["_Date"] = coerce_datetime(df[COL_TIME]).dt.date
df["_Created"] = df[COL_CREATED].astype(str).replace({"nan":"Unknown"})
if COL_ACCEPT is not None:
    df["_Marketing"] = df[COL_ACCEPT].apply(normalize_marketing)
else:
    df["_Marketing"] = "Unknown"

df["_Name"] = df.apply(build_name, axis=1)
if "Email" not in df.columns:
    df["Email"] = ""
if "Phone Number" not in df.columns:
    df["Phone Number"] = ""

# Search index (ensure columns exist)
SEARCH_COLS = ["_Name","Email","Phone Number","Company","Location","Title"]
for c in SEARCH_COLS:
    if c not in df.columns and c not in df:
        df[c] = ""

# -----------------------
# Header / top bar
# -----------------------
st.markdown(
    """
    <style>
    .topbar {display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;}
    .title {font-size:22px; font-weight:700;}
    .kpi-card {border:1px solid #1f2937; padding:14px 16px; border-radius:12px; background: var(--background-color, #0b1220);}
    .kpi-label {font-size:12px; color:#9ca3af; margin-bottom:6px;}
    .kpi-value {font-size:22px; font-weight:700;}
    .box {border:1px solid #1f2937; border-radius:12px; padding:16px; background: var(--background-color, #0b1220);}
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Responsive spacing & font sizes */
    @media (max-width: 992px) {
      .kpi-value { font-size: 18px !important; }
      .kpi-label { font-size: 11px !important; }
    }
    @media (max-width: 600px) {
      .kpi-value { font-size: 16px !important; }
      .kpi-label { font-size: 10px !important; }
      .topbar .title { font-size: 18px !important; }
    }
    /* Make Streamlit containers a touch tighter on mobile */
    section.main > div { padding-top: 0.5rem; }
    </style>
    """, unsafe_allow_html=True
)

colA, colB = st.columns([3,1])
with colA:
    st.markdown('<div class="topbar"><div class="title">Contacts Dashboard</div></div>', unsafe_allow_html=True)
with colB:
    q = st.text_input("🔎 Search…", key="q", placeholder="Search name, email, phone, company…")

# -----------------------
# Filters (horizontal)
# -----------------------
users = ["All Users"] + sorted(df["_Created"].fillna("Unknown").unique().tolist())
time_options = ["Last 7 days","Last 30 days","Last 90 days","This month","Last month","All time","Custom range"]
marketing_opts = ["All","Accepted","Declined","Unknown"]

c1, c2, c3, c4 = st.columns([1.2,1.2,1,1])
with c1:
    sel_user = st.selectbox("Created By", options=users, index=0)
with c2:
    sel_time = st.selectbox("Time Range", options=time_options, index=5)
with c3:
    sel_mkt = st.selectbox("Marketing", options=marketing_opts, index=0)
with c4:
    period_anchor = st.selectbox("Period Anchor", options=["Today (server time)", "Latest date in data"], index=0,
                                 help="Controls how preset ranges (Last 7/30/90 days, This/Last month) are computed.")
st.caption("Tip: The charts and table reflect your filters. Choose **All time** to include every row.")

# Custom range picker rendered only if chosen
custom_start, custom_end = None, None
if sel_time == "Custom range":
    cst, cen = st.columns(2)
    with cst:
        custom_start = st.date_input("Start date", value=df["_Date"].min() or date.today()-timedelta(days=30))
    with cen:
        custom_end = st.date_input("End date", value=df["_Date"].max() or date.today())

# -----------------------
# Apply filters
# -----------------------
filtered = df.copy()

# Time filter
if period_anchor == "Latest date in data" and not df["_Date"].dropna().empty:
    anchor_date = max(df["_Date"].dropna())
else:
    anchor_date = date.today()
if sel_time == "Last 7 days":
    start = anchor_date - timedelta(days=7)
    filtered = filtered[filtered["_Date"] >= start]
elif sel_time == "Last 30 days":
    start = anchor_date - timedelta(days=30)
    filtered = filtered[filtered["_Date"] >= start]
elif sel_time == "Last 90 days":
    start = anchor_date - timedelta(days=90)
    filtered = filtered[filtered["_Date"] >= start]
elif sel_time == "This month":
    start = date(anchor_date.year, anchor_date.month, 1)
    filtered = filtered[filtered["_Date"] >= start]
elif sel_time == "Last month":
    first_this = date(anchor_date.year, anchor_date.month, 1)
    last_month_end = first_this - timedelta(days=1)
    start = date(last_month_end.year, last_month_end.month, 1)
    filtered = filtered[(filtered["_Date"] >= start) & (filtered["_Date"] <= last_month_end)]
elif sel_time == "Custom range" and custom_start and custom_end:
    filtered = filtered[(filtered["_Date"] >= custom_start) & (filtered["_Date"] <= custom_end)]
# else "All time" -> no filter

# Created By filter
if sel_user != "All Users":
    filtered = filtered[filtered["_Created"] == sel_user]

# Marketing filter
if sel_mkt != "All":
    filtered = filtered[filtered["_Marketing"] == sel_mkt]

# Search filter
if q:
    ql = q.lower().strip()
    mask = pd.Series(False, index=filtered.index)
    for c in SEARCH_COLS:
        mask |= filtered[c].astype(str).str.lower().str.contains(ql, na=False)
    filtered = filtered[mask]


CHART_HEIGHT = 360

# -----------------------
# KPI row
# -----------------------
total_contacts = len(filtered)
marketing_accepted = int((filtered["_Marketing"] == "Accepted").sum())
marketing_declined = int((filtered["_Marketing"] == "Declined").sum())
active_users = filtered["_Created"].nunique()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Contacts</div><div class="kpi-value">{total_contacts:,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Marketing Accepted</div><div class="kpi-value">{marketing_accepted:,}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Marketing Declined</div><div class="kpi-value">{marketing_declined:,}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Active Users</div><div class="kpi-value">{active_users:,}</div></div>', unsafe_allow_html=True)

st.write("")
st.caption(f"Using **{period_anchor}** as {anchor_date:%b %d, %Y}.")

# -----------------------
# Charts row
# -----------------------
c_left, c_right = st.columns(2)


# Pie: Contacts by User
with c_left:
    counts = filtered["_Created"].fillna("Unknown").astype(str).value_counts().reset_index()
    counts.columns = ["Created By User", "Count"]
    counts["Label"] = counts["Created By User"] + " (" + counts["Count"].astype(str) + ")"
    fig_pie = px.pie(
        counts,
        values="Count",
        names="Label",
        hole=0.35,
        title="Contacts by User",
    )
    # Show values and percent on the slices; legend uses the Label with counts
    fig_pie.update_traces(textinfo="label+value+percent", hovertemplate="%{label}<br>Count: %{value} (%{percent})<extra></extra>")
    st.plotly_chart(fig_pie, use_container_width=True)


# Bar: Contacts over Time
with c_right:
    by_day = filtered.dropna(subset=["_Date"]).groupby("_Date").size().reset_index(name="Contacts Added")
    by_day = by_day.sort_values("_Date")
    fig_bar = px.bar(by_day, x="_Date", y="Contacts Added", title="Contacts Over Time")
fig_bar.update_layout(height=CHART_HEIGHT)
st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------
# Table (scrollable) + export
st.markdown("#### Contact Records")
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Export CSV", csv, file_name="contacts_filtered.csv", mime="text/csv")
st.dataframe(display.reset_index(drop=True), use_container_width=True, hide_index=True, height=520)
