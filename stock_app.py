import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import norm

# -----------------------------
# Polygon API setup
# -----------------------------
API_KEY = st.secrets["polygon"]["api_key"] 
BASE = "https://api.polygon.io"

def get_stock_data(ticker, start, end):
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    r = requests.get(url, params={"apiKey": API_KEY}).json()
    if "results" not in r: return pd.DataFrame()
    df = pd.DataFrame(r["results"])
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"c":"Close"}).set_index("t")
    return df

# -----------------------------
# Black-Scholes Option Pricing
# -----------------------------
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0: return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def estimate_volatility(stock_data, lookback=30):
    hist_returns = stock_data['Close'].pct_change().dropna()
    sigma = hist_returns[-lookback:].std() * np.sqrt(252)
    return sigma if sigma > 0 else 0.2

# -----------------------------
# Strategy Evaluation
# -----------------------------
def synthetic_premium(strat, S, K, T, r, sigma):
    if "Call" in strat:
        return black_scholes_call(S, K, T, r, sigma)
    else:
        return black_scholes_put(S, K, T, r, sigma)

def evaluate_option_trade(strat, spot, strike, premium, spot_at_exp):
    if "Buy Call" in strat:  return max(spot_at_exp - strike, 0) - premium
    if "Sell Call" in strat: return premium - max(spot_at_exp - strike, 0)
    if "Buy Put" in strat:   return max(strike - spot_at_exp, 0) - premium
    if "Sell Put" in strat:  return premium - max(strike - spot_at_exp, 0)
    return 0

def backtest_strategy(ticker, strat, expiry_offset, date_range, strike_mode="ATM"):
    all_trades = []
    r = 0.03
    start, end = date_range
    stock_data = get_stock_data(ticker, str(start), str(end + timedelta(days=expiry_offset)))
    if stock_data.empty: return pd.DataFrame()

    trade_dates = stock_data.index
    for d in trade_dates:
        spot = float(stock_data.loc[d, "Close"])

        expiry_idx = stock_data.index.get_indexer([d + timedelta(days=expiry_offset)], method="ffill")[0]
        if expiry_idx == -1: continue
        expiry = stock_data.index[expiry_idx]

        T = (expiry - d).days / 365
        sigma = estimate_volatility(stock_data)

        if strike_mode == "ATM":
            strike = spot
        elif strike_mode == "OTM":
            strike = spot * (0.98 if "Put" in strat else 1.02)
        elif strike_mode == "ITM":
            strike = spot * (1.02 if "Put" in strat else 0.98)

        premium = synthetic_premium(strat, spot, strike, T, r, sigma)
        spot_at_exp = float(stock_data.loc[expiry, "Close"])
        pnl = evaluate_option_trade(strat, spot, strike, premium, spot_at_exp)

        all_trades.append({
            "Entry Date": d,
            "Expiry": expiry,
            "Spot at Entry": spot,
            "Strike": strike,
            "Premium": premium,
            "Spot at Exp": spot_at_exp,
            "PnL": pnl
        })

    df_trades = pd.DataFrame(all_trades)
    if not df_trades.empty:
        df_trades["CumulativePnL"] = df_trades["PnL"].cumsum()
        df_trades = df_trades.groupby("Expiry").last()
    return df_trades

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Synthetic Options Backtester (Polygon.io)")

ticker = st.text_input("Ticker", "IWM")

col1, col2 = st.columns(2)
with col1:
    stratA = st.selectbox("Strategy A", ["Buy Call", "Sell Call", "Buy Put", "Sell Put"])
    expiryA = st.number_input("Expiry Offset (days) - A", 0, 90, 0)
    strikeA = st.selectbox("Strike Selection - A", ["ATM", "OTM", "ITM"])
with col2:
    stratB = st.selectbox("Strategy B", ["Buy Call", "Sell Call", "Buy Put", "Sell Put"])
    expiryB = st.number_input("Expiry Offset (days) - B", 0, 90, 30)
    strikeB = st.selectbox("Strike Selection - B", ["ATM", "OTM", "ITM"])

start_date = st.date_input("Start Date", datetime(2025,1,1))
end_date = st.date_input("End Date", datetime.today())

if st.button("Run Backtest"):
    dfA = backtest_strategy(ticker, stratA, expiryA, (start_date,end_date), strikeA)
    dfB = backtest_strategy(ticker, stratB, expiryB, (start_date,end_date), strikeB)

    if dfA.empty or dfB.empty:
        st.error("No trades generated. Try adjusting parameters or stock symbol.")
    else:
        # -----------------------------
        # Candlestick-style plot
        # -----------------------------
        def plot_candlestick(df, name, color):
            fig = go.Figure()
            for rid, df_range in enumerate(df.groupby(df.index)):
                df_r = df_range[1].copy()
                if df_r.empty: continue
                df_r["Open"] = df_r["CumulativePnL"].shift(1).fillna(0)
                fig.add_trace(go.Candlestick(
                    x=df_r.index,
                    open=df_r["Open"],
                    high=df_r["CumulativePnL"].cummax(),
                    low=df_r["CumulativePnL"].cummin(),
                    close=df_r["CumulativePnL"],
                    increasing_line_color=color,
                    decreasing_line_color=color,
                    name=f"{name} Range {rid+1}"
                ))
            fig.update_layout(title=f"Candlestick Cumulative PnL: {name}", template="plotly_white")
            return fig

        st.subheader("Candlestick-style Equity Curve")
        st.plotly_chart(plot_candlestick(dfA, stratA, "blue"), use_container_width=True)
        st.plotly_chart(plot_candlestick(dfB, stratB, "red"), use_container_width=True)

        # -----------------------------
        # Aligned line plot with difference
        # -----------------------------
        combined_index = dfA.index.union(dfB.index)
        dfA_plot = dfA.reindex(combined_index).ffill()
        dfB_plot = dfB.reindex(combined_index).ffill()
        diff = dfA_plot["CumulativePnL"] - dfB_plot["CumulativePnL"]

        st.subheader("Aligned Cumulative PnL Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfA_plot.index, y=dfA_plot["CumulativePnL"],
                                 mode="lines+markers", name=stratA, line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=dfB_plot.index, y=dfB_plot["CumulativePnL"],
                                 mode="lines+markers", name=stratB, line=dict(color="red")))
        fig.add_trace(go.Scatter(x=diff.index, y=diff.values,
                                 mode="lines+markers", name="Difference", line=dict(color="black", dash="dot")))
        fig.update_layout(
            title="Cumulative PnL Difference",
            xaxis_title="Date",
            yaxis_title="Cumulative PnL",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Full trade log + CSV export
        # -----------------------------
        dfA_full = dfA.reset_index()
        dfA_full["Strategy"] = stratA
        dfB_full = dfB.reset_index()
        dfB_full["Strategy"] = stratB
        df_all = pd.concat([dfA_full, dfB_full], ignore_index=True)

        st.subheader("All Trades (Full Data)")
        st.dataframe(df_all)

        csv = df_all.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Trade Log as CSV",
            data=csv,
            file_name=f"{ticker}_options_backtest.csv",
            mime="text/csv",
        )

        # -----------------------------
        # Line-style plot per trade
        # -----------------------------
        st.subheader("Individual Trade PnL Curves (All Trades)")
        fig2 = go.Figure()
        for strat, df in [(stratA, dfA_full), (stratB, dfB_full)]:
            df_sorted = df.sort_values("Expiry")
            df_sorted["CumulativePnL"] = df_sorted["PnL"].cumsum()
            fig2.add_trace(go.Scatter(
                x=df_sorted["Expiry"],
                y=df_sorted["CumulativePnL"],
                mode="lines+markers",
                name=strat
            ))
        fig2.update_layout(
            title="Cumulative PnL Per Trade",
            xaxis_title="Expiry",
            yaxis_title="Cumulative PnL",
            template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)
