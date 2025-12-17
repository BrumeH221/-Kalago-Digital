# app.py
from __future__ import annotations

import datetime as dt
import io
from typing import List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Reporting Dashboard: Revenue, Customers, Services",
    layout="wide",
    initial_sidebar_state="expanded",
)

TITLE = "📊 Reporting Dashboard: Revenue, Customers, Services"
ID_COLS = ["Client", "Job", "Job description", "Data type"]


# ----------------------------
# Theme + UI styling (SYNCED + BORDER FIX)
# ----------------------------
def _apply_theme(mode: str) -> None:
    """
    Modes:
      - Auto  : follow Streamlit theme (no forcing); consistent by design
      - Dark  : force dark AND override Streamlit CSS variables (!important)
      - Light : force light AND override Streamlit CSS variables (!important)

    Fixes:
      - Light mode widgets/tables not turning light
      - Missing text when switching theme
      - Border misalignment (use shadow-border, not border)
    """

    # Variables:
    # - Streamlit uses: --background-color, --secondary-background-color, --text-color, --primary-color
    # - We also define app vars: --app-bg, --app-card, --app-fg, --app-sidebar, --app-border, --app-shadow

    if mode == "Auto":
        vars_css = """
        <style>
          :root{
            --app-bg: var(--background-color);
            --app-card: var(--secondary-background-color);
            --app-sidebar: var(--secondary-background-color);
            --app-fg: var(--text-color);
            --app-border: rgba(120,120,120,0.35);
            --app-shadow: 0 10px 26px rgba(0,0,0,0.10);
            --app-pill-bg: rgba(120,120,120,0.16);
            --app-pill-fg: var(--text-color);
          }
        </style>
        """
    elif mode == "Dark":
        vars_css = """
        <style>
          html { color-scheme: dark; }
          :root{
            /* Force Streamlit theme vars */
            --primary-color: #60a5fa !important;
            --background-color: #0e1117 !important;
            --secondary-background-color: #111827 !important;
            --text-color: #f8fafc !important;

            /* App vars */
            --app-bg: #0e1117;
            --app-card: #111827;
            --app-sidebar: #0b1220;
            --app-fg: #f8fafc;

            --app-border: rgba(255,255,255,0.22);
            --app-shadow: 0 12px 30px rgba(0,0,0,0.35);

            --app-pill-bg: rgba(96,165,250,0.22);
            --app-pill-fg: #f8fafc;
          }
        </style>
        """
    else:  # Light
        vars_css = """
        <style>
          html { color-scheme: light; }
          :root{
            /* Force Streamlit theme vars */
            --primary-color: #2563eb !important;
            --background-color: #ffffff !important;
            --secondary-background-color: #ffffff !important;
            --text-color: #0f172a !important;

            /* App vars */
            --app-bg: #ffffff;
            --app-card: #ffffff;
            --app-sidebar: #f8fafc;
            --app-fg: #0f172a;

            --app-border: rgba(2,6,23,0.22);
            --app-shadow: 0 10px 26px rgba(2,6,23,0.08);

            --app-pill-bg: rgba(37,99,235,0.14);
            --app-pill-fg: #0f172a;
          }
        </style>
        """

    # Main CSS:
    # IMPORTANT: avoid global "span{color:...}" which can make some Streamlit widget text/icons disappear.
    css = """
    <style>
      * { box-sizing: border-box; }

      /* App background */
      .stApp {
        background: var(--app-bg) !important;
        color: var(--app-fg) !important;
      }
      .block-container { padding-top: 1.2rem; }

      /* Sidebar */
      section[data-testid="stSidebar"],
      section[data-testid="stSidebar"] > div {
        background: var(--app-sidebar) !important;
      }
      section[data-testid="stSidebar"]{
        box-shadow: 1px 0 0 0 var(--app-border);
      }

      /* --- Metric cards --- */
      .metric-grid{
        display:grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
      }
      @media (max-width: 1100px){
        .metric-grid{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }
      .metric-card{
        background: var(--app-card) !important;
        border-radius: 14px;
        padding: 14px 16px;
        /* Shadow-border: no layout shift */
        box-shadow: 0 0 0 1px var(--app-border), var(--app-shadow);
      }
      .metric-label{
        font-size: 12px;
        margin-bottom: 6px;
        opacity: 0.72;
        color: var(--app-fg) !important;
      }
      .metric-value{
        font-size: 22px;
        font-weight: 800;
        line-height: 1.1;
        color: var(--app-fg) !important;
      }
      .metric-help{
        font-size: 12px;
        margin-top: 4px;
        opacity: 0.72;
        color: var(--app-fg) !important;
      }

      /* --- Panels (charts + tables) --- */
      /* Charts: padding 12 looks good */
      div[data-testid="stVegaLiteChart"],
      div[data-testid="stAltairChart"]{
        background: var(--app-card) !important;
        border-radius: 12px;
        overflow: hidden;
        padding: 12px;
        box-shadow: 0 0 0 1px var(--app-border), var(--app-shadow);
      }

      /* Dataframe: NO padding (prevents “border looks offset”) */
      div[data-testid="stDataFrame"]{
        background: var(--app-card) !important;
        border-radius: 12px;
        overflow: hidden;
        padding: 0px;
        box-shadow: 0 0 0 1px var(--app-border), var(--app-shadow);
      }

      /* Dataframe internals: force background only (do NOT force all text colors globally) */
      div[data-testid="stDataFrame"] [data-baseweb="table"],
      div[data-testid="stDataFrame"] [role="grid"],
      div[data-testid="stDataFrame"] [role="rowgroup"],
      div[data-testid="stDataFrame"] [role="row"],
      div[data-testid="stDataFrame"] [role="columnheader"],
      div[data-testid="stDataFrame"] [role="gridcell"]{
        background: var(--app-card) !important;
      }

      /* --- Widgets --- */
      /* File uploader card */
      div[data-testid="stFileUploader"] section{
        background: var(--app-card) !important;
        border-radius: 12px !important;
        box-shadow: 0 0 0 1px var(--app-border) !important;
      }

      /* Selectbox / multiselect baseweb container */
      div[data-baseweb="select"] > div{
        background: var(--app-card) !important;
        box-shadow: 0 0 0 1px var(--app-border) !important;
      }

      /* Multiselect tags */
      span[data-baseweb="tag"]{
        background: var(--app-pill-bg) !important;
        color: var(--app-pill-fg) !important;
      }

      /* Buttons border */
      .stButton button{
        box-shadow: 0 0 0 1px var(--app-border) !important;
      }

      /* Tabs label visibility */
      button[data-baseweb="tab"]{
        color: var(--app-fg) !important;
      }
    </style>
    """

    st.markdown(vars_css + css, unsafe_allow_html=True)


def _enable_altair_theme(mode: str) -> None:
    """
    Auto: use Streamlit built-in Altair theme if available (Altair v5)
    Dark/Light: custom theme with visible plot frame and readable axes.
    """
    if mode == "Auto":
        try:
            alt.themes.enable("streamlit")
        except Exception:
            pass
        return

    is_dark = (mode == "Dark")

    def _dash_theme():
        if is_dark:
            axis_color = "#E5E7EB"
            grid_color = "rgba(255,255,255,0.18)"
            frame_color = "rgba(255,255,255,0.30)"
            title_color = "#F8FAFC"
        else:
            axis_color = "#0F172A"
            grid_color = "rgba(2, 6, 23, 0.16)"
            frame_color = "rgba(2, 6, 23, 0.26)"
            title_color = "#0F172A"

        return {
            "config": {
                "background": "transparent",
                "view": {"stroke": frame_color, "strokeWidth": 1},  # inner plot frame
                "title": {"color": title_color, "fontSize": 14},
                "axis": {
                    "labelColor": axis_color,
                    "titleColor": axis_color,
                    "gridColor": grid_color,
                    "tickColor": grid_color,
                    "domainColor": frame_color,
                },
                "legend": {"labelColor": axis_color, "titleColor": axis_color},
            }
        }

    try:
        alt.themes.register("dash_theme_synced_v3", _dash_theme)
    except Exception:
        pass
    alt.themes.enable("dash_theme_synced_v3")


def metric_cards(revenue: float, n_clients: int, n_services: int, n_jobs: int, currency: str) -> None:
    def fmt_money(x: float) -> str:
        return f"{currency}{x:,.2f}"

    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Total revenue</div>
            <div class="metric-value">{fmt_money(revenue)}</div>
            <div class="metric-help">Sum of Value within current filters</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Number of customers</div>
            <div class="metric-value">{n_clients:,}</div>
            <div class="metric-help">Clients with revenue &gt; 0</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Number of services</div>
            <div class="metric-value">{n_services:,}</div>
            <div class="metric-help">Distinct Job descriptions</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Number of jobs</div>
            <div class="metric-value">{n_jobs:,}</div>
            <div class="metric-help">Distinct Job IDs</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Data loading & shaping
# ----------------------------
@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    return pd.read_excel(bio, sheet_name=sheet_name)


def detect_month_cols(df: pd.DataFrame) -> List[pd.Timestamp]:
    month_cols: List[pd.Timestamp] = []
    for c in df.columns:
        if isinstance(c, (pd.Timestamp, dt.datetime, dt.date)):
            month_cols.append(pd.to_datetime(c))
    return sorted(month_cols)


def to_long(df_wide: pd.DataFrame) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    df = df_wide.copy()

    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    month_cols = detect_month_cols(df)
    if not month_cols:
        raise ValueError("No monthly datetime columns detected in Excel header. Please check the format.")

    value_cols = [c for c in df.columns if isinstance(c, (pd.Timestamp, dt.datetime, dt.date))]

    df_long = df.melt(
        id_vars=ID_COLS,
        value_vars=value_cols,
        var_name="Month",
        value_name="Value",
    )

    df_long["Month"] = pd.to_datetime(df_long["Month"]).dt.to_period("M").dt.to_timestamp()
    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce").fillna(0.0)

    months = sorted(df_long["Month"].unique())
    return df_long, months


def month_slider(min_m: pd.Timestamp, max_m: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    min_d = min_m.to_pydatetime().date().replace(day=1)
    max_d = max_m.to_pydatetime().date().replace(day=1)

    start_d, end_d = st.sidebar.slider(
        "Time range (monthly)",
        min_value=min_d,
        max_value=max_d,
        value=(min_d, max_d),
        format="YYYY-MM",
    )
    return pd.Timestamp(start_d), pd.Timestamp(end_d)


def multiselect_all(label: str, options: List[str], key: str) -> List[str]:
    """
    If user clears selection => treat as ALL (no filtering).
    """
    selected = st.sidebar.multiselect(label, options=options, default=options, key=key)
    return selected or options


# ----------------------------
# UI
# ----------------------------
st.title(TITLE)

with st.sidebar:
    st.header("⚙️ Filters & Options")

    mode = st.radio("Theme", ["Auto", "Dark", "Light"], index=0)
    _apply_theme(mode)
    _enable_altair_theme(mode)

    st.caption(
        "Tip: If you want Auto to follow Windows/Chrome: Streamlit menu (top-right) → "
        "Settings → Theme → Use system setting."
    )

    st.divider()
    st.write("Upload an Excel file or use the default file in `data/` folder.")
    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])


# Load default if no upload
default_path = "data/data.xlsx"
file_bytes: Optional[bytes] = None

if uploaded is not None:
    file_bytes = uploaded.getvalue()
else:
    try:
        with open(default_path, "rb") as f:
            file_bytes = f.read()
    except Exception:
        file_bytes = None

if file_bytes is None:
    st.info("No data file found. Please upload your Excel file from the sidebar.")
    st.stop()

# Sheet selector
xls = pd.ExcelFile(io.BytesIO(file_bytes))
sheets = xls.sheet_names
with st.sidebar:
    sheet = st.selectbox("Sheet", sheets, index=0)

df_wide = load_excel(file_bytes, sheet)
df_long, months = to_long(df_wide)

min_month = min(months)
max_month = max(months)

with st.sidebar:
    start_m, end_m = month_slider(min_month, max_month)

    currency = st.selectbox("Currency symbol", ["£", "$", "€", "₫"], index=0)

    data_types = sorted(df_long["Data type"].dropna().astype(str).unique().tolist())
    clients = sorted(df_long["Client"].dropna().astype(str).unique().tolist())
    services = sorted(df_long["Job description"].dropna().astype(str).unique().tolist())

    selected_types = multiselect_all("Data type", data_types, key="types")
    selected_clients = multiselect_all("Client", clients, key="clients")
    selected_services = multiselect_all("Service (Job description)", services, key="services")

    top_n = st.slider("Top N", min_value=5, max_value=30, value=10, step=1)

# Filter
mask = (
    (df_long["Month"] >= start_m)
    & (df_long["Month"] <= end_m)
    & (df_long["Data type"].astype(str).isin(selected_types))
    & (df_long["Client"].astype(str).isin(selected_clients))
    & (df_long["Job description"].astype(str).isin(selected_services))
)
dff = df_long.loc[mask].copy()

st.caption(
    f"Current filters: {start_m:%Y-%m} → {end_m:%Y-%m} · "
    f"Data type: {', '.join(selected_types[:3])}{'…' if len(selected_types) > 3 else ''} · "
    f"Clients: {', '.join(selected_clients[:3])}{'…' if len(selected_clients) > 3 else ''} · "
    f"Services: {', '.join(selected_services[:3])}{'…' if len(selected_services) > 3 else ''}"
)

if dff.empty:
    st.warning("No data found for the selected time range / filters.")
    st.stop()

# Metrics
revenue_total = float(dff["Value"].sum())
client_rev = dff.groupby("Client", as_index=False)["Value"].sum()
n_clients = int((client_rev["Value"] > 0).sum())
n_services = int(dff["Job description"].nunique())
n_jobs = int(dff["Job"].nunique())

metric_cards(revenue_total, n_clients, n_services, n_jobs, currency)

st.divider()
tabs = st.tabs(["Overview", "Customers", "Services", "Raw data"])


# ----------------------------
# Charts
# ----------------------------
def line_revenue_by_month(df_: pd.DataFrame) -> alt.Chart:
    m = df_.groupby("Month", as_index=False)["Value"].sum()
    return (
        alt.Chart(m)
        .mark_line(point=True)
        .encode(
            x=alt.X("Month:T", title="Month"),
            y=alt.Y("Value:Q", title="Revenue"),
            tooltip=[
                alt.Tooltip("Month:T", title="Month"),
                alt.Tooltip("Value:Q", title="Revenue", format=",.2f"),
            ],
        )
        .properties(height=320)
    )


def top_services_usage(df_: pd.DataFrame, n: int) -> alt.Chart:
    tmp = df_.loc[df_["Value"] > 0].copy()
    if tmp.empty:
        tmp = df_.copy()

    g = (
        tmp.groupby("Job description", as_index=False)
        .agg(Usage=("Job", "nunique"), Revenue=("Value", "sum"))
        .sort_values(["Usage", "Revenue"], ascending=False)
        .head(n)
    )

    return (
        alt.Chart(g)
        .mark_bar()
        .encode(
            y=alt.Y("Job description:N", sort="-x", title="Service"),
            x=alt.X("Usage:Q", title="Usage (unique jobs)"),
            tooltip=[
                alt.Tooltip("Job description:N", title="Service"),
                alt.Tooltip("Usage:Q", title="Usage"),
                alt.Tooltip("Revenue:Q", title="Revenue", format=",.2f"),
            ],
        )
        .properties(height=min(520, 28 * max(5, len(g))))
    )


# ----------------------------
# Tab: Overview
# ----------------------------
with tabs[0]:
    col1, col2 = st.columns([1.25, 1], gap="large")

    with col1:
        st.subheader("Monthly revenue trend")
        st.altair_chart(line_revenue_by_month(dff), use_container_width=True)

    with col2:
        st.subheader(f"Top {top_n} customers by revenue")
        top_clients = (
            dff.groupby("Client", as_index=False)["Value"]
            .sum()
            .sort_values("Value", ascending=False)
            .head(top_n)
        )
        top_clients.insert(0, "Rank", np.arange(1, len(top_clients) + 1))
        top_clients = top_clients.rename(columns={"Value": "Revenue"})
        st.dataframe(
            top_clients,
            use_container_width=True,
            hide_index=True,
            column_config={"Revenue": st.column_config.NumberColumn("Revenue", format=f"{currency}%,.2f")},
        )

    st.subheader(f"Top {top_n} services by usage")
    if dff.loc[dff["Value"] > 0].empty:
        st.info("No services with Value > 0 under current filters.")
    else:
        st.altair_chart(top_services_usage(dff, top_n), use_container_width=True)


# ----------------------------
# Tab: Customers
# ----------------------------
with tabs[1]:
    st.subheader("Customer ranking by revenue")
    cust = (
        dff.groupby("Client", as_index=False)["Value"]
        .sum()
        .sort_values("Value", ascending=False)
    )
    cust.insert(0, "Rank", np.arange(1, len(cust) + 1))
    cust = cust.rename(columns={"Value": "Revenue"})
    st.dataframe(
        cust,
        use_container_width=True,
        hide_index=True,
        column_config={"Revenue": st.column_config.NumberColumn("Revenue", format=f"{currency}%,.2f")},
    )


# ----------------------------
# Tab: Services
# ----------------------------
with tabs[2]:
    st.subheader("Service ranking")

    pos = dff.loc[dff["Value"] > 0].copy()
    if pos.empty:
        pos = dff.copy()

    service_stats = (
        pos.groupby("Job description", as_index=False)
        .agg(
            Revenue=("Value", "sum"),
            Usage=("Job", "nunique"),
            Jobs=("Job", "nunique"),
            Customers=("Client", "nunique"),
        )
        .sort_values(["Usage", "Revenue"], ascending=False)
    )
    service_stats.insert(0, "Rank", np.arange(1, len(service_stats) + 1))

    st.dataframe(
        service_stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Revenue": st.column_config.NumberColumn("Revenue", format=f"{currency}%,.2f"),
            "Usage": st.column_config.NumberColumn("Usage", help="Distinct Job IDs with Value > 0"),
        },
    )


# ----------------------------
# Tab: Raw data
# ----------------------------
with tabs[3]:
    st.subheader("Filtered long-format data")
    st.dataframe(dff, use_container_width=True, hide_index=True)

    csv = dff.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv",
    )
