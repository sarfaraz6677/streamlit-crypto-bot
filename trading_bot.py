# --- IMPORTS ---
import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from streamlit_option_menu import option_menu

# --- STREAMLIT CONFIG ---
st.set_page_config(layout='wide')
st.title("🚀 SarfaraJ & Faisal's Trading Bot (BTC/USDT)")

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'adjustForTimeDifference': True}
})
symbol = 'BTC/USDT'

st.markdown(f"<div style='font-size:24px; font-weight:bold; color:green;'>🕒 Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div>", unsafe_allow_html=True)

# --- DATA FETCH ---
@st.cache_data(ttl=300)
def fetch_data(timeframe, limit):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Add Indicators
        df['rsi'] = RSIIndicator(df['close']).rsi()
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['sma_14'] = SMAIndicator(df['close'], window=14).sma_indicator()
        df['ema_14'] = EMAIndicator(df['close'], window=14).ema_indicator()
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

        return df.dropna()
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

def prepare_features(df):
    df['target'] = np.where(df['close'].shift(-3) > df['close'], 1, 0)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['rsi_diff'] = df['rsi'] - 50
    df['price_change'] = df['close'] - df['open']
    features = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'rsi_diff', 'price_change', 'sma_14', 'ema_14', 'atr']
    return df.dropna(), features

@st.cache_resource
def train_all_models(df):
    df, features = prepare_features(df)
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.25)

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=300, max_depth=10, min_child_samples=20, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=10, eval_metric='logloss', random_state=42)
    }

    trained_models = {}
    accuracies = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        trained_models[name] = model
        accuracies[name] = acc

    return trained_models, features, accuracies

# --- PREDICTION FUNCTION ---
def predict_latest(model, df, features):
    X_latest = df[features].iloc[-1:]
    proba = model.predict_proba(X_latest)[0][1]
    pred = model.predict(X_latest)[0]
    return pred, proba

# --- MASHWARA ---
def get_mashwara(signal, prob, mode):
    if signal == 1 and prob > 0.65:
        msg = f"{mode} signal kehta hai BUY karo! (Confidence: {prob:.2f})"
    elif signal == 1:
        msg = f"{mode} signal BUY ki taraf hai, magar confidence kam hai ({prob:.2f})."
    else:
        msg = f"{mode} signal SELL keh raha hai. (Confidence: {prob:.2f})"
    return f"<p style='font-size:22px; font-weight:bold; color:#2E86C1;'>{msg}</p>"

# --- CHART ---
def draw_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], mode='lines', name='RSI'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], mode='lines', name='MACD Signal'))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    return fig

# --- BACKTEST ---
def realistic_backtest(df, model, features, initial_balance=1000):
    df = df.copy()
    df['prediction'] = model.predict(df[features])
    balance = initial_balance
    position = None
    trades = []
    for i in range(len(df)-1):
        pred = df['prediction'].iloc[i]
        price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        slippage = 0.001
        fee = 0.001

        if pred == 1 and not position:
            entry = price * (1 + slippage)
            stop = entry - atr * 1.5
            target = entry + atr * 3
            position = 'long'

        elif position == 'long':
            close = price
            if close <= stop or close >= target or pred == 0:
                pnl = (close - entry) - (entry * fee) - (close * fee)
                balance += pnl
                trades.append(pnl)
                position = None

    profit = balance - initial_balance
    wins = [t for t in trades if t > 0]
    accuracy = (len(wins) / len(trades)) * 100 if trades else 0
    return profit, accuracy, len(trades), balance

# --- FETCH DATA ---
sc_df = fetch_data('5m', 1000)
sw_df = fetch_data('1h', 1000)

# --- TRAIN MODELS ---
sc_models, sc_features, sc_accuracies = train_all_models(sc_df) if sc_df is not None else ({}, [], {})
sw_models, sw_features, sw_accuracies = train_all_models(sw_df) if sw_df is not None else ({}, [], {})

# --- MODEL SELECTOR ---
st.markdown("")
selected_model_name = option_menu(
    menu_title=None,
    options=["RandomForest", "LightGBM", "XGBoost"],
    icons=["tree", "lightbulb", "fire"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "rgb(14 17 23)"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {
            "font-size": "18px",
            "text-align": "center",
            "margin": "0px 10px",
            "--hover-color": "rgba(255, 165, 0, 0.3)",
            "color": "white",
        },
        "nav-link-selected": {
            "background-color": "transparent",
            "color": "white",
            "font-weight": "bold",
            "text-shadow": "0 0 8px orange",
        },
    }
)

# --- SCALPING PREDICTION ---
if sc_df is not None and selected_model_name in sc_models:
    sc_df_prepared, _ = prepare_features(sc_df)
    sc_model = sc_models[selected_model_name]
    sc_pred, sc_prob = predict_latest(sc_model, sc_df_prepared, sc_features)
    sc_profit, sc_acc, sc_trades, sc_bal = realistic_backtest(sc_df_prepared, sc_model, sc_features)
else:
    sc_pred, sc_prob, sc_profit, sc_acc, sc_trades, sc_bal = 0, 0.0, 0, 0, 0, 0.0

# --- SWING PREDICTION ---
if sw_df is not None and selected_model_name in sw_models:
    sw_df_prepared, _ = prepare_features(sw_df)
    sw_model = sw_models[selected_model_name]
    sw_pred, sw_prob = predict_latest(sw_model, sw_df_prepared, sw_features)
    sw_profit, sw_acc, sw_trades, sw_bal = realistic_backtest(sw_df_prepared, sw_model, sw_features)
else:
    sw_pred, sw_prob, sw_profit, sw_acc, sw_trades, sw_bal = 0, 0.0, 0, 0, 0, 0.0

# --- DISPLAY SIDE BY SIDE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Scalping (5-minute)")
    st.markdown(get_mashwara(sc_pred, sc_prob, "Scalping"), unsafe_allow_html=True)
    st.write(f"Backtest Profit: ${sc_profit:.2f}")
    st.write(f"Backtest Accuracy: {sc_acc:.2f}% on {sc_trades} trades")
    if sc_df is not None:
        st.plotly_chart(draw_chart(sc_df, "BTC/USDT Scalping Chart"), use_container_width=True)

with col2:
    st.subheader("Swing (1-hour)")
    st.markdown(get_mashwara(sw_pred, sw_prob, "Swing"), unsafe_allow_html=True)
    st.write(f"Backtest Profit: ${sw_profit:.2f}")
    st.write(f"Backtest Accuracy: {sw_acc:.2f}% on {sw_trades} trades")
    if sw_df is not None:
        st.plotly_chart(draw_chart(sw_df, "BTC/USDT Swing Chart"), use_container_width=True)
