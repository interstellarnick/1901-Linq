
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, date, timedelta

st.set_page_config(page_title="Contacts Dashboard", layout="wide")

# -----------------------
# Helpers
# -----------------------
@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    return pd.read_csv("data/admin_contacts.csv")

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

# Column mapping
COL_TIME = next((c for c in df.columns if c.lower() == "time" or "date" in c.lower()), None)
COL_CREATED = next((c for c in df.columns if c.lower().strip() == "created by user"), None) or \
              next((c for c in df.columns if "created by" in c.lower()), None)
COL_ACCEPT = next((c for c in df.columns if "accept marketing" in c.lower()), None)

if COL_TIME is None or COL_CREATED is None:
    st.error("Could not infer required columns. Expected 'Time' (or date) and 'Created By User'.")
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

# Search index (lowercased)
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
    .searchbox {min-width:280px;}
    .kpi-card {border:1px solid #1f2937; padding:14px 16px; border-radius:12px; background: #0b1220;}
    .kpi-label {font-size:12px; color:#9ca3af; margin-bottom:6px;}
    .kpi-value {font-size:22px; font-weight:700;}
    .tag {padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid #1f2937; display:inline-block;}
    .tag-accepted {background:#052e16; color:#a7f3d0; border-color:#065f46;}
    .tag-declined {background:#450a0a; color:#fecaca; border-color:#7f1d1d;}
    .tag-unknown {background:#111827; color:#d1d5db; border-color:#374151;}
    .box {border:1px solid #1f2937; border-radius:12px; padding:16px; background:#0b1220;}
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

c1, c2, c3, c4 = st.columns([1.3,1.3,1,0.8])
with c1:
    sel_user = st.selectbox("Created By", options=users, index=0)
with c2:
    sel_time = st.selectbox("Time Range", options=time_options, index=1)
with c3:
    sel_mkt = st.selectbox("Marketing", options=marketing_opts, index=0)
with c4:
    apply = st.button("Apply Filters", use_container_width=True, type="primary")

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
today = date.today()
if sel_time == "Last 7 days":
    start = today - timedelta(days=7)
    filtered = filtered[filtered["_Date"] >= start]
elif sel_time == "Last 30 days":
    start = today - timedelta(days=30)
    filtered = filtered[filtered["_Date"] >= start]
elif sel_time == "Last 90 days":
    start = today - timedelta(days=90)
    filtered = filtered[filtered["_Date"] >= start]
elif sel_time == "This month":
    start = date(today.year, today.month, 1)
    filtered = filtered[filtered["_Date"] >= start]
elif sel_time == "Last month":
    first_this = date(today.year, today.month, 1)
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

# -----------------------
# Charts row
# -----------------------
c_left, c_right = st.columns(2)

# Pie: Contacts by User
with c_left:
    counts = filtered["_Created"].fillna("Unknown").astype(str).value_counts().reset_index()
    counts.columns = ["Created By User", "Count"]
    fig_pie = px.pie(counts, values="Count", names="Created By User", hole=0.35, title="Contacts by User")
    st.plotly_chart(fig_pie, use_container_width=True)

# Bar: Contacts over Time
with c_right:
    by_day = filtered.dropna(subset=["_Date"]).groupby("_Date").size().reset_index(name="Contacts Added")
    by_day = by_day.sort_values("_Date")
    fig_bar = px.bar(by_day, x="_Date", y="Contacts Added", title="Contacts Over Time")
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------
# Table with pagination + export
# -----------------------
st.markdown("#### Contact Records")

display = filtered.copy()
display["Date"] = pd.to_datetime(display["_Date"], errors="coerce").dt.strftime("%b %d, %Y")
display["Created By"] = display["_Created"]

# Marketing badge text
def badge_txt(v):
    if v == "Accepted":
        return "Accepted"
    if v == "Declined":
        return "Declined"
    return "Unknown"

display["Marketing"] = display["_Marketing"].apply(badge_txt)

table_cols = ["_Name","Phone Number","Email","Created By","Date","Marketing"]
display = display[table_cols].rename(columns={"_Name":"Name"})

# Pagination
PAGE_SIZE = 10
total_pages = max(1, int(np.ceil(len(display) / PAGE_SIZE)))
page = st.session_state.get("page", 1)

prev_col, pages_col, next_col, export_col = st.columns([0.6, 2, 0.6, 1])
with prev_col:
    if st.button("◀ Previous", use_container_width=True, disabled=(page<=1)):
        page = max(1, page-1)
with pages_col:
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=page, step=1, label_visibility="collapsed")
with next_col:
    if st.button("Next ▶", use_container_width=True, disabled=(page>=total_pages)):
        page = min(total_pages, page+1)
with export_col:
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", csv, file_name="contacts_filtered.csv", mime="text/csv", use_container_width=True)

st.session_state["page"] = int(page)

start = (page-1)*PAGE_SIZE
end = start + PAGE_SIZE
st.dataframe(display.iloc[start:end].reset_index(drop=True), use_container_width=True, hide_index=True)

st.caption(f"Showing {start+1} to {min(end, len(display))} of {len(display)} results")
