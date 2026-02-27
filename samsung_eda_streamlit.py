"""
Samsung Global Sales — Streamlit EDA App
=========================================
Run with:
    pip install streamlit pandas numpy plotly seaborn matplotlib
    streamlit run samsung_eda_streamlit.py
"""
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Samsung Global Sales EDA",
                   page_icon="📱", layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #050a14; }
    [data-testid="stSidebar"] { background-color: #0b1627; }
    [data-testid="metric-container"] {
        background: #0b1627;
        border: 1px solid #1a2d4d;
        border-radius: 4px;
        padding: 12px;
    }
    h1, h2, h3 { color: #e8f0ff !important; }
    p, label, .stMarkdown { color: #6b85a8 !important; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

COLORS = px.colors.qualitative.Bold

# ── Load data ─────────────────────────────────────────────────────────────────


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["sale_date"])
    df["year"] = df["year"].astype(str)
    df["month_num"] = pd.to_datetime(df["sale_date"]).dt.month
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📱 Samsung EDA")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    st.markdown("---")
    if uploaded:
        df_raw = pd.read_csv(uploaded, parse_dates=["sale_date"])
        df_raw["month_num"] = pd.to_datetime(df_raw["sale_date"]).dt.month
        df_raw["year"] = df_raw["year"].astype(str)
    else:
        st.info("Upload `samsung_global_sales_dataset.csv` to begin.")
        st.stop()

    st.markdown("### 🔧 Filters")
    years = st.multiselect("Year", sorted(
        df_raw["year"].unique()), default=sorted(df_raw["year"].unique()))
    regions = st.multiselect("Region", sorted(
        df_raw["region"].unique()), default=sorted(df_raw["region"].unique()))
    categories = st.multiselect("Category", sorted(
        df_raw["category"].unique()), default=sorted(df_raw["category"].unique()))

    st.markdown("---")
    st.markdown("### 📊 Navigation")
    page = st.radio("Section", [
        "🏠 Overview",
        "📦 Product Analysis",
        "🌍 Regional Analysis",
        "👤 Customer Analysis",
        "🛒 Sales Channels",
        "🔬 Data Quality",
    ])

# ── Apply Filters ─────────────────────────────────────────────────────────────
df = df_raw[
    df_raw["year"].isin(years) &
    df_raw["region"].isin(regions) &
    df_raw["category"].isin(categories)
].copy()

if df.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  🏠 OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📱 Samsung Global Sales — EDA Dashboard")
    st.markdown(
        f"Analysing **{len(df):,}** transactions · **{df['year'].nunique()}** years · **{df['region'].nunique()}** regions · **{df['category'].nunique()}** categories")
    st.markdown("---")

    # KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(label="💰 Total Revenue",
              value=f"${df['revenue_usd'].sum()/1e6:.2f}M")
    k2.metric(label="🧾 Transactions", value=f"{len(df):,}")
    k3.metric(label="📦 Avg Order Value",
              value=f"${df['revenue_usd'].mean():,.0f}")
    k4.metric(label="↩️ Return Rate",
              value=f"{(df['return_status'] == 'Returned').mean()*100:.1f}%")
    k5.metric(label="⭐ Avg Rating",
              value=f"{df['customer_rating'].mean():.2f}")
    k6.metric(label="🏷️ Avg Discount",
              value=f"{df['discount_pct'].mean():.1f}%")
    st.markdown("---")

    # Revenue by Year
    col1, col2 = st.columns(2)
    # Revenue by Year
    with col1:
        st.subheader("Revenue by Year")
        yr = df.groupby("year")["revenue_usd"].sum().reset_index()
        fig = px.bar(yr, x="year", y="revenue_usd", color="year",
                     labels={"revenue_usd": "Revenue (USD)", "year": "Year"},
                     color_discrete_sequence=COLORS, template="plotly_dark")
        fig.update_layout(showlegend=False,
                          plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    # Revenue by Quarter
    with col2:
        st.subheader("Revenue by Quarter (aggregated)")
        qtr = df.groupby("quarter")["revenue_usd"].sum().reset_index()
        fig = px.bar(qtr, x="quarter", y="revenue_usd", color="quarter",
                     labels={"revenue_usd": "Revenue (USD)"},
                     color_discrete_sequence=COLORS, template="plotly_dark")
        fig.update_layout(showlegend=False,
                          plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

     # Monthly Trend
    st.subheader("Monthly Revenue Trend")
    monthly = df.groupby(["year", "month_num"])[
        "revenue_usd"].sum().reset_index()
    monthly["period"] = monthly["year"].astype(
        str) + "-" + monthly["month_num"].astype(str).str.zfill(2)
    fig = px.line(monthly, x="period", y="revenue_usd", color="year",
                  labels={"revenue_usd": "Revenue (USD)", "period": "Month"},
                  color_discrete_sequence=COLORS, template="plotly_dark",
                  markers=True)
    fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

    # Revenue & Units heatmap: Year x Quarter
    st.subheader("Revenue Heatmap — Year × Quarter")
    heat = df.groupby(["year", "quarter"])["revenue_usd"].sum().reset_index()
    heat_pivot = heat.pivot(
        index="year", columns="quarter", values="revenue_usd")
    fig = px.imshow(heat_pivot, text_auto=".2s", color_continuous_scale="Blues",
                    template="plotly_dark", aspect="auto")
    fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

    # Descriptive stats
    st.subheader("📋 Descriptive Statistics")
    num_cols = ["unit_price_usd", "discount_pct",
                "units_sold", "revenue_usd", "customer_rating"]
    st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  📦 PRODUCT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Product Analysis":
    st.title("📦 Product Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by Category")
        cat = df.groupby("category")["revenue_usd"].sum(
        ).sort_values(ascending=True).reset_index()
        fig = px.bar(cat, x="revenue_usd", y="category", orientation="h",
                     color="category", color_discrete_sequence=COLORS, template="plotly_dark",
                     labels={"revenue_usd": "Revenue (USD)", "category": "Category"})
        fig.update_layout(showlegend=False,
                          plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Transaction Count by Category")
        cat_cnt = df["category"].value_counts().reset_index()
        # st.dataframe(cat_cnt)
        cat_cnt.columns = ["category", "count"]
        fig = px.pie(cat_cnt, names="category", values="count",
                     color_discrete_sequence=COLORS, template="plotly_dark", hole=0.45)
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    # Top products
    st.subheader("Top 15 Products by Revenue")
    top = df.groupby("product_name")["revenue_usd"].sum(
    ).sort_values(ascending=False).head(15).reset_index()
    fig = px.bar(top, x="revenue_usd", y="product_name", orientation="h",
                 color="revenue_usd", color_continuous_scale="Blues", template="plotly_dark",
                 labels={"revenue_usd": "Revenue (USD)", "product_name": "Product"})
    fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627", yaxis={
                      "autorange": "reversed"})
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("5G vs Non-5G Revenue")
        fg = df.groupby("is_5g")["revenue_usd"].sum().reset_index()
        fig = px.pie(fg, names="is_5g", values="revenue_usd", hole=0.5,
                     color_discrete_sequence=["#1565ff", "#00c6ff"], template="plotly_dark")
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Avg Rating by Category")
        rat = df.groupby("category")["customer_rating"].mean(
        ).sort_values(ascending=True).reset_index()
        fig = px.bar(rat, x="customer_rating", y="category", orientation="h",
                     color="customer_rating", color_continuous_scale="Tealgrn",
                     template="plotly_dark", range_x=[3.5, 4.0])
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    # Discount distribution
    st.subheader("Discount % Distribution by Category")
    fig = px.box(df, x="category", y="discount_pct", color="category",
                 color_discrete_sequence=COLORS, template="plotly_dark",
                 labels={"discount_pct": "Discount (%)", "category": "Category"})
    fig.update_layout(showlegend=False, plot_bgcolor="#0b1627",
                      paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

    # Revenue vs Units scatter
    st.subheader("Unit Price vs Revenue (by Category)")
    fig = px.scatter(df, x="unit_price_usd", y="revenue_usd", color="category",
                     size="units_sold", hover_data=["product_name"],
                     color_discrete_sequence=COLORS, template="plotly_dark", opacity=0.6,
                     labels={"unit_price_usd": "Unit Price (USD)", "revenue_usd": "Revenue (USD)"})
    fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  🌍 REGIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Regional Analysis":
    st.title("🌍 Regional Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by Region")
        reg = df.groupby("region")["revenue_usd"].sum(
        ).sort_values(ascending=True).reset_index()
        fig = px.bar(reg, x="revenue_usd", y="region", orientation="h",
                     color="region", color_discrete_sequence=COLORS, template="plotly_dark")
        fig.update_layout(showlegend=False,
                          plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Region Share (Pie)")
        fig = px.pie(reg, names="region", values="revenue_usd", hole=0.4,
                     color_discrete_sequence=COLORS, template="plotly_dark")
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    # Top countries
    st.subheader("Top 20 Countries by Revenue")
    country = df.groupby("country")["revenue_usd"].sum(
    ).sort_values(ascending=False).head(20).reset_index()
    fig = px.bar(country, x="country", y="revenue_usd", color="revenue_usd",
                 color_continuous_scale="Blues", template="plotly_dark",
                 labels={"revenue_usd": "Revenue (USD)", "country": "Country"})
    fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

    # Region × Category heatmap
    st.subheader("Revenue Heatmap — Region × Category")
    rc = df.groupby(["region", "category"])["revenue_usd"].sum().reset_index()
    rc_pivot = rc.pivot(index="region", columns="category",
                        values="revenue_usd").fillna(0)
    fig = px.imshow(rc_pivot, text_auto=".2s", color_continuous_scale="Blues",
                    template="plotly_dark", aspect="auto")
    fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

    # Region trend over years
    st.subheader("Regional Revenue Trend by Year")
    ry = df.groupby(["region", "year"])["revenue_usd"].sum().reset_index()
    fig = px.line(ry, x="year", y="revenue_usd", color="region",
                  markers=True, color_discrete_sequence=COLORS, template="plotly_dark",
                  labels={"revenue_usd": "Revenue (USD)", "year": "Year"})
    fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  👤 CUSTOMER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Customer Analysis":
    st.title("👤 Customer Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Segments")
        seg = df["customer_segment"].value_counts().reset_index()
        seg.columns = ["segment", "count"]
        fig = px.pie(seg, names="segment", values="count", hole=0.45,
                     color_discrete_sequence=COLORS, template="plotly_dark")
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Age Group Distribution")
        age = df["customer_age_group"].value_counts().reset_index()
        age.columns = ["age_group", "count"]
        fig = px.bar(age, x="age_group", y="count", color="age_group",
                     color_discrete_sequence=COLORS, template="plotly_dark")
        fig.update_layout(showlegend=False,
                          plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    # Revenue by segment
    st.subheader("Revenue by Customer Segment × Year")
    seg_yr = df.groupby(["customer_segment", "year"])[
        "revenue_usd"].sum().reset_index()
    fig = px.bar(seg_yr, x="year", y="revenue_usd", color="customer_segment",
                 barmode="group", color_discrete_sequence=COLORS, template="plotly_dark",
                 labels={"revenue_usd": "Revenue (USD)"})
    fig.update_layout(showlegend=False,
                      plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Return Status Breakdown")
        ret = df["return_status"].value_counts().reset_index()
        ret.columns = ["status", "count"]
        fig = px.pie(ret, names="status", values="count", hole=0.5,
                     color_discrete_sequence=["#4ade80", "#f87171", "#f5a623"],
                     template="plotly_dark")
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Customer Rating Distribution")
        fig = px.histogram(df.dropna(subset=["customer_rating"]), x="customer_rating",
                           nbins=20, color_discrete_sequence=["#1565ff"],
                           template="plotly_dark", labels={"customer_rating": "Rating"})
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

# Previous device OS
    st.subheader("Previous Device OS (top sources)")
    os_df = df["previous_device_os"].dropna().value_counts().reset_index()
    os_df.columns = ["os", "count"]
    fig = px.bar(os_df, x="os", y="count", color="os",
                 color_discrete_sequence=COLORS, template="plotly_dark")
    fig.update_layout(showlegend=False, plot_bgcolor="#0b1627",
                      paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

    # Rating by age group
    st.subheader("Avg Rating by Age Group × Category")

    fig = px.density_heatmap(df.dropna(subset=["customer_rating"]), x="category", y="customer_age_group", z="customer_rating", histfunc="avg",   # automatically calculates mean
                             text_auto=".2f", color_continuous_scale="RdBu", template="plotly_dark")
    fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627",
                      coloraxis_colorbar_title="Avg Rating")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  🛒 SALES CHANNELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛒 Sales Channels":
    st.title("🛒 Sales Channels & Payments")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by Sales Channel")
        ch = df.groupby("sales_channel")["revenue_usd"].sum(
        ).sort_values(ascending=True).reset_index()
        fig = px.bar(ch, x="revenue_usd", y="sales_channel", orientation="h",
                     color="sales_channel", color_discrete_sequence=COLORS, template="plotly_dark")
        fig.update_layout(showlegend=False,
                          plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Payment Method Distribution")
        pay = df["payment_method"].value_counts().reset_index()
        pay.columns = ["method", "count"]
        fig = px.pie(pay, names="method", values="count", hole=0.4,
                     color_discrete_sequence=COLORS, template="plotly_dark")
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    # Channel × Category Revenue Heatmap (Without Pivot)
    st.subheader("Channel × Category Revenue Heatmap")
    cc = df.groupby(["sales_channel", "category"])[
        "revenue_usd"].sum().reset_index()
    fig = px.density_heatmap(cc, x="category", y="sales_channel", z="revenue_usd",
                             text_auto=".2s", color_continuous_scale="Blues", template="plotly_dark")
    fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

    # Channel trend
    st.subheader("Sales Channel Trend by Year")
    cy = df.groupby(["sales_channel", "year"])[
        "revenue_usd"].sum().reset_index()
    fig = px.line(cy, x="year", y="revenue_usd", color="sales_channel",
                  markers=True, color_discrete_sequence=COLORS, template="plotly_dark")
    fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Avg Order Value by Channel")
        aov = df.groupby("sales_channel")["revenue_usd"].mean(
        ).sort_values(ascending=True).reset_index()
        fig = px.bar(aov, x="revenue_usd", y="sales_channel", orientation="h",
                     color="revenue_usd", color_continuous_scale="Blues", template="plotly_dark",
                     labels={"revenue_usd": "Avg Revenue (USD)"})
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Payment Method × Segment")
        ps = df.groupby(["payment_method", "customer_segment"])[
            "revenue_usd"].sum().reset_index()
        fig = px.bar(ps, x="payment_method", y="revenue_usd", color="customer_segment",
                     barmode="stack", color_discrete_sequence=COLORS, template="plotly_dark",
                     labels={"revenue_usd": "Revenue (USD)"})
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627",
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  🔬 DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Data Quality":
    st.title("🔬 Data Quality & Raw Data")
    st.markdown("---")

    st.subheader('Missing Values')
    miss = df.isnull().sum().reset_index()
    miss.columns = ["column", "missing"]
    miss["pct"] = (miss["missing"] / len(df) * 100).round(2)
    miss = miss[miss["missing"] > 0].sort_values(
        "missing", ascending=False)
    if miss.empty:
        st.success("✅ No missing values in current filtered data.")
    else:
        fig = px.bar(miss, x="column", y="pct", color="pct",
                     color_continuous_scale="Reds", template="plotly_dark",
                     labels={"pct": "Missing %", "column": "Column"})
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(miss, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Matrix (Numeric Features)")
    num = df[["unit_price_usd", "discount_pct", "units_sold",
              "discounted_price_usd", "revenue_usd", "customer_rating"]].dropna()
    corr = num.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                    template="plotly_dark", color_continuous_midpoint=0)
    fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Outlier Detection — Revenue Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, y="revenue_usd", color="category",
                     color_discrete_sequence=COLORS, template="plotly_dark",
                     labels={"revenue_usd": "Revenue (USD)"})
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="revenue_usd", nbins=50,
                           color_discrete_sequence=["#1565ff"], template="plotly_dark",
                           labels={"revenue_usd": "Revenue (USD)"})
        fig.update_layout(plot_bgcolor="#0b1627", paper_bgcolor="#0b1627")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📄 Raw Data Explorer")
    st.dataframe(df.head(500), use_container_width=True)

    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="samsung_filtered.csv",
        mime="text/csv"
    )
