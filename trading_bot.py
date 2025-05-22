# --- IMPORTS ---
import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from streamlit_option_menu import option_menu
import hashlib

# --- STREAMLIT CONFIG ---
st.set_page_config(layout='wide')
st.title("🚀 Sarfaraz & Faisal's Trading Bot (BTC/USDT)")

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'adjustForTimeDifference': True}
})
symbol = 'BTC/USDT'

st.markdown(
    f"<div style='font-size:24px; font-weight:bold; color:green;'>🕒 Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div>",
    unsafe_allow_html=True
)

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

        # Additional Features
        df['momentum'] = df['close'] - df['close'].shift(1)
        df['volatility'] = df['high'] - df['low']

        return df.dropna()
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

def prepare_features(df):
    df['target'] = np.where(df['close'].shift(-3) > df['close'], 1, 0)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['rsi_diff'] = df['rsi'] - 50
    df['price_change'] = df['close'] - df['open']
    features = [
        'rsi', 'macd', 'macd_signal', 'macd_hist', 'rsi_diff',
        'price_change', 'sma_14', 'ema_14', 'atr', 'momentum', 'volatility'
    ]
    return df.dropna(), features

def hash_df(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

@st.cache_resource(show_spinner=False)
def train_all_models(df_hash, df):
    df, features = prepare_features(df)
    X = df[features]
    y = df['target']
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=400, max_depth=15, min_child_samples=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=400, max_depth=15, eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1)
    }

    trained_models = {}
    accuracies = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=30, verbose=False)
        except TypeError:
            model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        trained_models[name] = model
        accuracies[name] = acc

    return trained_models, features, accuracies

def predict_latest(model, df, features):
    X_latest = df[features].iloc[-1:]
    proba = model.predict_proba(X_latest)[0][1]
    pred = model.predict(X_latest)[0]
    return pred, proba

def get_mashwara(signal, prob, mode):
    if signal == 1 and prob > 0.65:
        msg = f"{mode} signal kehta hai BUY karo! (Confidence: {prob:.2f})"
        color = "#28a745"
    elif signal == 1:
        msg = f"{mode} signal BUY ki taraf hai, magar confidence kam hai ({prob:.2f})."
        color = "#ffc107"
    else:
        msg = f"{mode} signal SELL keh raha hai. (Confidence: {prob:.2f})"
        color = "#dc3545"
    return f"<p style='font-size:22px; font-weight:bold; color:{color};'>{msg}</p>"

def draw_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], mode='lines', name='RSI'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], mode='lines', name='MACD Signal'))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    return fig

def realistic_backtest(df, model, features, initial_balance=1000):
    df = df.copy()
    df['prediction'] = model.predict(df[features])
    balance = initial_balance
    position = None
    trades = []

    for i in range(len(df) - 1):
        pred = df['prediction'].iloc[i]
        price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        slippage = 0.001
        fee = 0.001

        if pred == 1 and position is None:
            entry_price = price * (1 + slippage)
            position = {
                'entry': entry_price,
                'stop': entry_price - atr * 1.5,
                'target': entry_price + atr * 3
            }

        elif position is not None:
            close_price = price
            exit_trade = False

            if close_price <= position['stop'] or close_price >= position['target'] or pred == 0:
                exit_trade = True

            if exit_trade:
                pnl = (close_price - position['entry']) - (position['entry'] * fee) - (close_price * fee)
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
sc_models, sc_features, sc_accuracies = {}, [], {}
sw_models, sw_features, sw_accuracies = {}, [], {}

if sc_df is not None:
    sc_df_hash = hash_df(sc_df)
    sc_models, sc_features, sc_accuracies = train_all_models(sc_df_hash, sc_df)

if sw_df is not None:
    sw_df_hash = hash_df(sw_df)
    sw_models, sw_features, sw_accuracies = train_all_models(sw_df_hash, sw_df)

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

# --- DISPLAY SIGNALS AND CHART ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"⚡ Scalping (5 Min) - Model: {selected_model_name}")
    if sc_df is not None:
        st.plotly_chart(draw_chart(sc_df, "Scalping View"), use_container_width=True)
        st.markdown(get_mashwara(sc_pred, sc_prob, f"Scalping ({selected_model_name})"), unsafe_allow_html=True)
        st.markdown(f"💰 **Final Balance:** ${sc_bal:.2f}")
        st.markdown(f"📈 **Profit:** ${sc_profit:.2f}")
        st.markdown(f"✅ **Accuracy:** {sc_acc:.2f}% over {sc_trades} trades")

with col2:
    st.subheader(f"📊 Swing Trading (1 Hour) - Model: {selected_model_name}")
    if sw_df is not None:
        st.plotly_chart(draw_chart(sw_df, "Swing View"), use_container_width=True)
        st.markdown(get_mashwara(sw_pred, sw_prob, f"Swing ({selected_model_name})"), unsafe_allow_html=True)
        st.markdown(f"💰 **Final Balance:** ${sw_bal:.2f}")
        st.markdown(f"📈 **Profit:** ${sw_profit:.2f}")
        st.markdown(f"✅ **Accuracy:** {sw_acc:.2f}% over {sw_trades} trades")

# --- TRANSPARENT TABLE STYLING ---
st.markdown("---")
st.subheader("📊 Model Accuracy Summary (Test Set)")

styled_df = (
    pd.DataFrame({
        'Model': list(sc_accuracies.keys()),
        'Scalping Accuracy (%)': [sc_accuracies.get(m, 0) * 100 for m in sc_accuracies.keys()],
        'Swing Accuracy (%)': [sw_accuracies.get(m, 0) * 100 for m in sc_accuracies.keys()],
    })
    .set_index('Model')
    .style
    .format("{:.2f}")
    .background_gradient(subset=['Scalping Accuracy (%)'], cmap='Greens')
    .background_gradient(subset=['Swing Accuracy (%)'], cmap='Purples')
    .set_properties(**{
        'font-size': '30px',
        'font-weight': 'bold',
        'border': '1px solid #444',
        'text-align': 'center',
        'background-color': 'transparent',   # <--- Transparent Background
        'color': '#fff',
    })
    .set_table_styles([
        {
            'selector': 'thead',
            'props': [
                ('background-color', 'rgba(255, 255, 255, 0.05)'),  # Semi-transparent head
                ('color', '#fff'),
                ('font-size', '30px'),
                ('font-weight', 'bold')
            ]
        },
        {
            'selector': 'th',
            'props': [('text-align', 'center')]
        }
    ])
)

st.dataframe(styled_df, use_container_width=True)

st.markdown(
    """
    <div style='font-size:20px; color:#bbb; margin-top:8px;'>
    💡 Accuracy is based on test set (last 1000 candles). Color intensity = model performance.
    </div>
    """, unsafe_allow_html=True
)


