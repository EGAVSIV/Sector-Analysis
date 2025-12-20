# =====================================================
# IMPORTS
# =====================================================
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import hashlib

# =====================================================
# STREAMLIT CONFIG (MUST BE FIRST)
# =====================================================
st.set_page_config(
    page_title="🧭 NIFTY Sector Rotation & Relative Strength",
    layout="wide",
)

# =====================================================
# LOGIN SYSTEM
# =====================================================
def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

USERS = st.secrets["users"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login Required")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in USERS and hash_pwd(p) == USERS[u]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# =====================================================
# PAGE HEADER
# =====================================================
st.title("📊 Sector Rotation & Relative Strength Dashboard")
st.caption("Sector Performance | RRG | Rotation | Portfolio | Scanner")

# =====================================================
# PATH CONFIG
# =====================================================
DATA_FOLDER = "sectorial_index_data"

STOCK_DATA_REPO = (
    "https://raw.githubusercontent.com/"
    "EGAVSIV/Stock_Scanner_With_ASTA_Parameters/main/stock_data_D"
)

STATE_FILE = "sector_rotation_state.csv"

# =====================================================
# DATA LOADERS
# =====================================================
@st.cache_data(show_spinner=False)
def load_sector_parquet(path):
    try:
        df = pd.read_parquet(path)

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        elif "datetime" in df.index.names:
            df = df.reset_index()

        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        return df.sort_values("date")

    except Exception as e:
        st.warning(f"⚠️ Sector load failed: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_stock_parquet(symbol):
    try:
        url = f"{STOCK_DATA_REPO}/{symbol}.parquet"
        df = pd.read_parquet(url)

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        elif "datetime" in df.index.names:
            df = df.reset_index()

        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        return df.sort_values("date")

    except Exception:
        return None

# =====================================================
# UTILITIES
# =====================================================
def calc_return(df, bars):
    if df is None or len(df) < bars:
        return None
    return (df["close"].iloc[-1] / df["close"].iloc[-bars] - 1) * 100


def classify_rotation(r1m, r3m):
    if r1m > 0 and r3m > 0:
        return "Leading"
    if r1m < 0 and r3m > 0:
        return "Weakening"
    if r1m < 0 and r3m < 0:
        return "Lagging"
    return "Improving"

# =====================================================
# LOAD NIFTY + LAST UPDATED DATE
# =====================================================
nifty_df_all = load_sector_parquet(os.path.join(DATA_FOLDER, "NIFTY.parquet"))
if nifty_df_all is None:
    st.error("❌ NIFTY data not available")
    st.stop()

last_updated = nifty_df_all["date"].iloc[-1]
st.success(f"🕒 Data Last Updated: {last_updated}")

# =====================================================
# UI CONTROL
# =====================================================
lookback = st.slider("Lookback Sessions", 10, 60, 30, step=5)

# =====================================================
# SECTOR VS NIFTY (UNCHANGED LOGIC)
# =====================================================
results = []
nifty_ret = calc_return(nifty_df_all.tail(lookback), lookback)

for f in os.listdir(DATA_FOLDER):
    if not f.endswith(".parquet") or f == "NIFTY.parquet":
        continue

    df = load_sector_parquet(os.path.join(DATA_FOLDER, f))
    r = calc_return(df.tail(lookback), lookback)

    if r is None:
        continue

    results.append({
        "Symbol": f.replace(".parquet", ""),
        "Return %": round(r, 2),
        "NIFTY %": round(nifty_ret, 2),
        "Status": "Outperforming" if r > nifty_ret else "Underperforming"
    })

df_result = pd.DataFrame(results).sort_values("Return %", ascending=False)

st.subheader("📋 Performance Table")
st.dataframe(df_result, width="stretch")

st.subheader("📈 Relative Performance Chart")
fig, ax = plt.subplots(figsize=(14, 7))
colors = ["green" if r > nifty_ret else "red" for r in df_result["Return %"]]
ax.bar(df_result["Symbol"], df_result["Return %"], color=colors)
ax.axhline(nifty_ret, color="blue", linestyle="--", label="NIFTY")
plt.xticks(rotation=45, ha="right")
plt.legend()
st.pyplot(fig)

# =====================================================
# MULTI-TIMEFRAME ROTATION
# =====================================================
rotation_rows = []

for f in os.listdir(DATA_FOLDER):
    if not f.endswith(".parquet") or f == "NIFTY.parquet":
        continue

    df = load_sector_parquet(os.path.join(DATA_FOLDER, f))
    r1m = calc_return(df, 21)
    r3m = calc_return(df, 63)
    r6m = calc_return(df, 126)

    if r1m is None or r3m is None:
        continue

    rotation_rows.append({
        "Sector": f.replace(".parquet", ""),
        "1M %": round(r1m, 2),
        "3M %": round(r3m, 2),
        "6M %": round(r6m, 2) if r6m else np.nan,
        "Rotation": classify_rotation(r1m, r3m)
    })

df_rotation = pd.DataFrame(rotation_rows)
df_rotation["RS Rank"] = df_rotation["1M %"].rank(ascending=False).astype(int)

st.header("🔁 Sector Rotation Table")
st.dataframe(df_rotation.sort_values("RS Rank"), width="stretch")

# =====================================================
# RRG STYLE SCATTER
# =====================================================
st.header("🧭 RRG-Style Sector Map")

rrg_df = df_rotation.copy()
rrg_df["Momentum"] = rrg_df["1M %"] - rrg_df["3M %"]

fig, ax = plt.subplots(figsize=(10, 8))
for _, r in rrg_df.iterrows():
    ax.scatter(r["1M %"], r["Momentum"], s=120)
    ax.text(r["1M %"], r["Momentum"], r["Sector"], fontsize=9)

ax.axhline(0, linestyle="--", color="grey")
ax.axvline(0, linestyle="--", color="grey")
ax.set_xlabel("Relative Strength (1M %)")
ax.set_ylabel("Momentum (1M − 3M)")
st.pyplot(fig)

# =====================================================
# TOP 5 SECTORS
# =====================================================
st.header("🏆 Top 5 Strongest Sectors")
st.dataframe(df_rotation.sort_values("RS Rank").head(5), width="stretch")

# =====================================================
# ROTATION CHANGE ALERTS
# =====================================================
st.header("🚨 Sector Rotation Change Alerts")

prev_state = {}
if os.path.exists(STATE_FILE):
    prev_state = dict(pd.read_csv(STATE_FILE).values)

alerts = []
for _, r in df_rotation.iterrows():
    old = prev_state.get(r["Sector"])
    if old and old != r["Rotation"]:
        alerts.append(f"{r['Sector']}: {old} → {r['Rotation']}")

df_rotation[["Sector", "Rotation"]].to_csv(STATE_FILE, index=False)

if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.success("No sector rotation changes.")

# =====================================================
# SECTOR → STOCK MAPPING
# =====================================================
SECTOR_STOCKS = {
    "CNXIT": ["TCS", "INFY", "WIPRO","HCL","LTIM","TECM","COFORGE","MPHASIS","ORACLE"],
    "CNXAUTO": ["TATAMOTORS", "MARUTI", "M&M"],
    "CNXFINANCE": ["HDFCBANK", "ICICIBANK", "AXISBANK"]
}

# =====================================================
# STOCK LEVEL RRG
# =====================================================
st.header("📈 Stock-Level RRG Inside Sector")

available_sectors = sorted(set(SECTOR_STOCKS) & set(df_rotation["Sector"]))

if available_sectors:
    sector_sel = st.selectbox("Select Sector", available_sectors)

    sector_df = load_sector_parquet(
        os.path.join(DATA_FOLDER, f"{sector_sel}.parquet")
    )
    sector_1m = calc_return(sector_df, 21)

    rows = []
    for stock in SECTOR_STOCKS.get(sector_sel, []):
        sdf = load_stock_parquet(stock)
        if sdf is None:
            continue

        r1 = calc_return(sdf, 21)
        r3 = calc_return(sdf, 63)

        if r1 is None or r3 is None:
            continue

        rows.append({
            "Stock": stock,
            "RS vs Sector": round(r1 - sector_1m, 2),
            "Momentum": round(r1 - r3, 2)
        })

    if rows:
        df_stock_rrg = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
        ax.scatter(
            df_stock_rrg["RS vs Sector"],
            df_stock_rrg["Momentum"],
            s=90,
            alpha=0.85,
            edgecolors="black"
        )
        for _, r in df_stock_rrg.iterrows():
            ax.annotate(
                r["Stock"],
                (r["RS vs Sector"], r["Momentum"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                weight="bold"
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axhspan(0, ax.get_ylim()[1], xmin=0.5, xmax=1, color="#e6ffe6", alpha=0.3)   # Leading
        ax.axhspan(0, ax.get_ylim()[1], xmin=0, xmax=0.5, color="#e6f0ff", alpha=0.3)   # Improving
        ax.axhspan(ax.get_ylim()[0], 0, xmin=0, xmax=0.5, color="#fff0e6", alpha=0.3)  # Lagging
        ax.axhspan(ax.get_ylim()[0], 0, xmin=0.5, xmax=1, color="#fff5f5", alpha=0.3)  # Weakening

        # Dynamic axis limits (tight view)
        x_pad = max(1, abs(df_stock_rrg["RS vs Sector"]).max() * 0.25)
        y_pad = max(1, abs(df_stock_rrg["Momentum"]).max() * 0.25)

        ax.set_xlim(
            df_stock_rrg["RS vs Sector"].min() - x_pad,
            df_stock_rrg["RS vs Sector"].max() + x_pad
        )
        ax.set_ylim(
            df_stock_rrg["Momentum"].min() - y_pad,
            df_stock_rrg["Momentum"].max() + y_pad
        )

        ax.set_xlabel("Relative Strength vs Sector (1M %)", fontsize=10)
        ax.set_ylabel("Momentum (1M − 3M)", fontsize=10)
        ax.set_title(f"Stock-Level RRG — {sector_sel}", fontsize=12, weight="bold")

        ax.grid(alpha=0.3)

        st.pyplot(fig)


# =====================================================
# SECTOR WEIGHTED PORTFOLIO
# =====================================================
st.header("🧮 Sector Weight-Adjusted Portfolio")

model = df_rotation[df_rotation["Rotation"].isin(["Leading", "Improving"])].copy()
model["Raw Weight"] = model["RS Rank"].max() - model["RS Rank"] + 1
model["Weight %"] = (model["Raw Weight"] / model["Raw Weight"].sum()) * 100

st.dataframe(
    model[["Sector", "Rotation", "RS Rank", "Weight %"]],
    width="stretch"
)

# =====================================================
# AUTO SECTOR-BASED STOCK SCANNER
# =====================================================
st.header("🤖 Auto Sector-Based Stock Scanner")

scanner = []
for _, r in model.iterrows():
    sector = r["Sector"]
    s_df = load_sector_parquet(os.path.join(DATA_FOLDER, f"{sector}.parquet"))
    s_ret = calc_return(s_df, 21)

    for stock in SECTOR_STOCKS.get(sector, []):
        sdf = load_stock_parquet(stock)
        if sdf is None:
            continue

        sr = calc_return(sdf, 21)
        if sr and sr > s_ret:
            scanner.append({
                "Sector": sector,
                "Stock": stock,
                "Stock 1M %": round(sr, 2),
                "Sector 1M %": round(s_ret, 2),
                "Signal": "Sector Leader"
            })

if scanner:
    st.dataframe(pd.DataFrame(scanner), width="stretch")
else:
    st.info("No sector-leading stocks detected.")

# =====================================================
# WEEKLY PDF EXPORT
# =====================================================
st.header("📄 Weekly Sector Rotation Report")

def make_pdf(df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=10)

    pdf.cell(
        0, 8,
        "Weekly Sector Rotation Report",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C"
    )
    pdf.ln(6)

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    for _, r in df.sort_values("RS Rank").iterrows():
        line = (
            f"{r['Sector']} | "
            f"1M: {r['1M %']}% | "
            f"3M: {r['3M %']}% | "
            f"{r['Rotation']}"
        )

        pdf.multi_cell(
            usable_width,
            6,
            line,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT
        )

    return bytes(pdf.output(dest="S"))

pdf_bytes = make_pdf(df_rotation)

st.download_button(
    "📥 Download Weekly PDF Report",
    pdf_bytes,
    "sector_rotation_weekly_report.pdf",
    "application/pdf"
)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
---
**Designed by Gaurav Singh Yadav**  
Sector Rotation | Relative Strength | Quant Intelligence  
""")
