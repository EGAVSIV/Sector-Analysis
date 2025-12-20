import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Sector Performance vs NIFTY",
    layout="wide",
)

st.title("📊 Sector Performance vs NIFTY")
st.caption("Last 30 Trading Sessions | Data Source: Parquet (GitHub Actions)")

DATA_FOLDER = "sectorial_index_data"

# ===============================
# DATA LOADER
# ===============================
@st.cache_data(show_spinner=False)
def load_parquet_data(file_path):
    try:
        df = pd.read_parquet(file_path)

        # Reset index safely
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        elif 'datetime' in df.index.names:
            df = df.reset_index()

        if 'datetime' not in df.columns:
            raise ValueError("Missing datetime column")

        df['date'] = pd.to_datetime(df['datetime']).dt.date
        return df

    except Exception as e:
        st.warning(f"⚠️ Error reading {os.path.basename(file_path)}: {e}")
        return None

# ===============================
# CALCULATION
# ===============================
def calculate_vs_nifty(folder_path, lookback=30):
    results = []

    nifty_path = os.path.join(folder_path, "NIFTY.parquet")
    nifty_df = load_parquet_data(nifty_path)

    if nifty_df is None or len(nifty_df) < lookback:
        st.error("❌ Not enough NIFTY data")
        return pd.DataFrame()

    nifty_df = nifty_df.sort_values("date").tail(lookback)
    nifty_return = (nifty_df['close'].iloc[-1] / nifty_df['close'].iloc[0] - 1) * 100

    for file in os.listdir(folder_path):
        if not file.endswith(".parquet") or file == "NIFTY.parquet":
            continue

        df = load_parquet_data(os.path.join(folder_path, file))
        if df is None or len(df) < lookback:
            continue

        df = df.sort_values("date").tail(lookback)
        stock_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100

        results.append({
            "Symbol": file.replace(".parquet", ""),
            "Return %": round(stock_return, 2),
            "NIFTY %": round(nifty_return, 2),
            "Status": "Outperforming" if stock_return > nifty_return else "Underperforming"
        })

    return pd.DataFrame(results).sort_values("Return %", ascending=False)

# ===============================
# PLOT
# ===============================
def plot_performance(df):
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ["green" if r > df["NIFTY %"].iloc[0] else "red" for r in df["Return %"]]
    bars = ax.bar(df["Symbol"], df["Return %"], color=colors)

    for bar, val in zip(bars, df["Return %"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.2f}%", ha='center', va='bottom', fontsize=9)

    nifty_val = df["NIFTY %"].iloc[0]
    ax.axhline(nifty_val, color="blue", linestyle="--", label=f"NIFTY ({nifty_val:.2f}%)")

    ax.set_ylabel("Return (%)")
    ax.set_title("Sector Performance vs NIFTY (Last 30 Sessions)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig

# ===============================
# UI CONTROLS
# ===============================
lookback = st.slider("Lookback Sessions", 10, 60, 30, step=5)

if not os.path.exists(DATA_FOLDER):
    st.error("❌ sectorial_index_data folder not found")
    st.stop()

df_result = calculate_vs_nifty(DATA_FOLDER, lookback)

if df_result.empty:
    st.warning("No results to display")
    st.stop()

# ===============================
# DISPLAY
# ===============================
st.subheader("📋 Performance Table")
st.dataframe(df_result, use_container_width=True)

st.subheader("📈 Relative Performance Chart")
fig = plot_performance(df_result)
st.pyplot(fig)

# ===============================
# DOWNLOAD
# ===============================
csv = df_result.to_csv(index=False)
st.download_button(
    "📥 Download CSV",
    csv,
    "sector_performance_vs_nifty.csv",
    "text/csv"
)

st.markdown("""
---
**Designed by Gaurav Singh Yadav**  
Sector Rotation | Relative Strength | Quant Analysis
""")
