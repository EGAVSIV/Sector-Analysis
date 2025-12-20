import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Sector Performance & Rotation Dashboard",
    layout="wide",
)

st.title("📊 Sector Performance vs NIFTY")
st.caption("Relative Strength | Sector Rotation | Multi-Timeframe Analysis")

DATA_FOLDER = "sectorial_index_data"

# ===============================
# DATA LOADER
# ===============================
@st.cache_data(show_spinner=False)
def load_parquet_data(file_path):
    df = pd.read_parquet(file_path)

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif 'datetime' in df.index.names:
        df = df.reset_index()

    df['date'] = pd.to_datetime(df['datetime']).dt.date
    return df.sort_values("date")

# ===============================
# LAST UPDATED DATE
# ===============================
nifty_df_all = load_parquet_data(os.path.join(DATA_FOLDER, "NIFTY.parquet"))
last_updated = nifty_df_all['date'].iloc[-1]

st.success(f"🕒 **Data Last Updated:** {last_updated}")

# ===============================
# BASIC VS NIFTY (KEEP AS IS)
# ===============================
def calculate_vs_nifty(folder_path, lookback=30):
    results = []

    nifty_df = nifty_df_all.tail(lookback)
    nifty_return = (nifty_df['close'].iloc[-1] / nifty_df['close'].iloc[0] - 1) * 100

    for file in os.listdir(folder_path):
        if not file.endswith(".parquet") or file == "NIFTY.parquet":
            continue

        df = load_parquet_data(os.path.join(folder_path, file))
        if len(df) < lookback:
            continue

        df = df.tail(lookback)
        stock_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100

        results.append({
            "Symbol": file.replace(".parquet", ""),
            "Return %": round(stock_return, 2),
            "NIFTY %": round(nifty_return, 2),
            "Status": "Outperforming" if stock_return > nifty_return else "Underperforming"
        })

    return pd.DataFrame(results).sort_values("Return %", ascending=False)

# ===============================
# MULTI TIMEFRAME PERFORMANCE
# ===============================
def multi_tf_returns(df):
    periods = {
        "1W %": 5,
        "1M %": 21,
        "3M %": 63,
        "6M %": 126,
    }
    out = {}
    for k, p in periods.items():
        if len(df) >= p:
            out[k] = round((df['close'].iloc[-1] / df['close'].iloc[-p] - 1) * 100, 2)
        else:
            out[k] = np.nan
    return out

# ===============================
# SECTOR ROTATION LOGIC
# ===============================
def classify_rotation(ret_1m, ret_3m):
    if ret_1m > 0 and ret_3m > 0:
        return "🟢 Leading"
    if ret_1m < 0 and ret_3m > 0:
        return "🟡 Weakening"
    if ret_1m < 0 and ret_3m < 0:
        return "🔴 Lagging"
    return "🔵 Improving"

# ===============================
# UI CONTROL
# ===============================
lookback = st.slider("Lookback Sessions (Primary Table)", 10, 60, 30, step=5)

df_result = calculate_vs_nifty(DATA_FOLDER, lookback)

# ===============================
# EXISTING DISPLAY (UNCHANGED)
# ===============================
st.subheader("📋 Performance Table (UNCHANGED)")
st.dataframe(df_result, use_container_width=True)

st.subheader("📈 Relative Performance Chart (UNCHANGED)")
fig, ax = plt.subplots(figsize=(14, 7))
colors = ["green" if r > df_result["NIFTY %"].iloc[0] else "red" for r in df_result["Return %"]]
ax.bar(df_result["Symbol"], df_result["Return %"], color=colors)
ax.axhline(df_result["NIFTY %"].iloc[0], color="blue", linestyle="--", label="NIFTY")
plt.xticks(rotation=45, ha="right")
plt.legend()
st.pyplot(fig)

# ===============================
# 🔥 NEW SECTION: SECTOR ROTATION
# ===============================
st.markdown("---")
st.header("🔁 Sector Rotation & Multi-Timeframe Strength")

rotation_rows = []

for file in os.listdir(DATA_FOLDER):
    if not file.endswith(".parquet") or file == "NIFTY.parquet":
        continue

    df = load_parquet_data(os.path.join(DATA_FOLDER, file))
    tf = multi_tf_returns(df)

    rotation = classify_rotation(tf["1M %"], tf["3M %"])

    rotation_rows.append({
        "Sector": file.replace(".parquet", ""),
        **tf,
        "Rotation": rotation
    })

df_rotation = pd.DataFrame(rotation_rows)

# RS Rank
df_rotation["RS Rank"] = df_rotation["1M %"].rank(ascending=False).astype(int)

st.subheader("🧭 Sector Rotation Table (NEW)")
st.dataframe(
    df_rotation.sort_values("RS Rank"),
    use_container_width=True
)

# ===============================
# DOWNLOADS
# ===============================
st.download_button(
    "📥 Download Rotation CSV",
    df_rotation.to_csv(index=False),
    "sector_rotation_analysis.csv",
    "text/csv"
)

st.markdown("""
---
**Designed by Gaurav Singh Yadav**  
Sector Rotation | Relative Strength | Quant Intelligence  
""")
