# Synthetic Options Backtester

This is a Streamlit app for backtesting simple options strategies using historical stock data from [Polygon.io](https://polygon.io/).

Developed as commission for personal use for a certain Senior Investment Analyst to play with.

Not sure if accurate, but fully functional.

You can simulate "Buy Call", "Sell Call", "Buy Put", and "Sell Put" strategies, visualize cumulative PnL, and download the full trade log.

---

## Features

- Fetch historical stock prices using Polygon.io API  
- Estimate volatility and price synthetic options using Black-Scholes model  
- Backtest strategies with customizable expiry offsets and strike types (ATM, OTM, ITM)  
- Candlestick-style PnL plots  
- Aligned PnL comparison of multiple strategies  
- Export full trade logs as CSV  

---

## Installation

1. Clone the repository:

```
git clone https://github.com/kaipokrandt/stockbacktestertool.git
cd stockbacktestertool
```
2. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## Setup Polygon.io API Key

Store your API key securely using Streamlit Secrets:

1. Create a `.streamlit/secrets.toml` file:

```
[polygon]
api_key = "YOUR_REAL_API_KEY"
```

2. The app will automatically read it with:

```
import streamlit as st
API_KEY = st.secrets["polygon"]["api_key"]
```

---

## Usage

Run the app locally:

```
streamlit run main.py
```

- Enter the stock ticker (e.g., `IWM`)  
- Select Strategy A & B, expiry offsets, and strike types  
- Choose start and end dates  
- Click **Run Backtest**  
- View candlestick and line plots  
- Download full trade logs  

---

## Security Warning

⚠️ Do **not** hardcode your Polygon.io API key if the repository is public. Use Streamlit Secrets or environment variables.  

---

## License

MIT License

---

## Acknowledgements

- [Polygon.io](https://polygon.io/) for historical stock data  
- [Streamlit](https://streamlit.io/) for the interactive UI  
- [Plotly](https://plotly.com/python/) for plotting charts  
