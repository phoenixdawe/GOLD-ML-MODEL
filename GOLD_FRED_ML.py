import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
FRED_API_KEY = "601b81ed717c38ac7dd912053b08c754"
fred = Fred(api_key=FRED_API_KEY)

START_DATE = "2018-01-01"
END_DATE = "2026-07-01" 

print("🤖 Safely downloading individual market vectors from Yahoo Finance...")

# Download each ticker individually to bypass yfinance NoneType dataframe parsing errors
sp500 = yf.download("^GSPC", start=START_DATE, end=END_DATE)[["Close"]].rename(columns={"Close": "^GSPC"})
tnx = yf.download("^TNX", start=START_DATE, end=END_DATE)[["Close"]].rename(columns={"Close": "^TNX"})
gold = yf.download("GC=F", start=START_DATE, end=END_DATE)[["Close"]].rename(columns={"Close": "GC=F"})

# Clean multi-index headers if present
for df_temp in [sp500, tnx, gold]:
    df_temp.columns = df_temp.columns.get_level_values(0) if isinstance(df_temp.columns, pd.MultiIndex) else df_temp.columns

# Combine Market Assets
market_df = pd.concat([sp500, tnx, gold], axis=1).ffill()

print("🏦 Extracting stable, validated economic series from FRED...")
# Replacing EXPINF5 with T5YIE (Official 5-Year Market Breakeven Inflation Rate)
fred_series = {
    "expected_inf_1y": "MICH",
    "expected_inf_5y": "T5YIE",
    "real_interest_rate": "DFII10"
}

fred_dfs = []
for name, series_id in fred_series.items():
    try:
        s_data = fred.get_series(series_id, start=START_DATE, end=END_DATE)
        fred_dfs.append(pd.DataFrame({name: s_data}))
    except Exception as e:
        print(f"⚠️ Error pulling FRED series ID {series_id}: {e}")

macro_data = pd.concat(fred_dfs, axis=1).ffill()

# Merge all into one Master Array
master_df = pd.concat([market_df, macro_data], axis=1).ffill().dropna()

# Check to ensure array size is sufficient for training mathematical components
if master_df.empty:
    raise ValueError("The merged dataset is empty. Check dates or data alignment.")

print("📈 Extracting engineered mathematical features...")
# Rate of Change (RoC)
master_df["sp500_roc"] = master_df["^GSPC"].pct_change(1)
master_df["us_10y_roc"] = master_df["^TNX"].pct_change(1)

# Rolling Z-Score for Gold Futures (GC=F) - 20 trading day baseline
rolling_mean = master_df["GC=F"].rolling(window=20).mean()
rolling_std = master_df["GC=F"].rolling(window=20).std()
master_df["gold_zscore"] = (master_df["GC=F"] - rolling_mean) / rolling_std

# Directional & Magnitude targets based on your exact 2-day lookahead instructions
master_df["target_direction"] = np.where(master_df["GC=F"].shift(-2) > master_df["GC=F"], 1, 0)
master_df["target_magnitude"] = master_df["GC=F"].shift(-2) - master_df["GC=F"]

# Clear edge calculations
master_df.dropna(inplace=True)

# Build feature dimensions
feature_cols = ["sp500_roc", "us_10y_roc", "gold_zscore", "expected_inf_1y", "expected_inf_5y", "real_interest_rate"]
X = master_df[feature_cols]
y_cls = master_df["target_direction"]
y_reg = master_df["target_magnitude"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. ENGINE MODEL TRAINING
# ==========================================
print("🧠 Training dual ML pipeline (Classifier + Regressor)...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_scaled, y_cls)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_scaled, y_reg)

# ==========================================
# 4. RECENT PREDICTION & TRADE EXECUTION LOGIC
# ==========================================
latest_features = X_scaled[-1].reshape(1, -1)
current_gold_price = float(master_df["GC=F"].iloc[-1])

# Extract probability scalar array values explicitly
prob_increase = float(clf.predict_proba(latest_features)[0][1])
expected_magnitude = float(reg.predict(latest_features)[0])

print("\n" + "="*45)
print(f"📊 LIVE MODEL METRICS FOR {master_df.index[-1].strftime('%Y-%m-%d')}")
print(f"Current Gold Price (GC=F): ${current_gold_price:,.2f}")
print(f"Calculated Probability of Increase: {prob_increase * 100:.2f}%")
print(f"Predicted Magnitude Change (2-Day Window): ${expected_magnitude:+.2f}")
print("="*45)

print("⚡ TRADE EXECUTION ENGINE EVALUATION:")
if prob_increase >= 0.60:
    target_price = current_gold_price + abs(expected_magnitude)
    stop_loss = current_gold_price - (abs(expected_magnitude) / 2)
    print("🟢 ACTION REQUIRED: [BUY/LONG]")
    print(f"   ↳ Aiming for Target Price: ${target_price:,.2f}")
    print(f"   ↳ Protective Stop-Loss Set: ${stop_loss:,.2f}")

elif prob_increase <= 0.40:
    target_price = current_gold_price - abs(expected_magnitude)
    stop_loss = current_gold_price + (abs(expected_magnitude) / 2)
    print("🔴 ACTION REQUIRED: [SHORT]")
    print(f"   ↳ Aiming for Target Price: ${target_price:,.2f}")
    print(f"   ↳ Protective Stop-Loss Set: ${stop_loss:,.2f}")

else:
    print("⚪ ACTION REQUIRED: [NO TRADE / HOLD]")
    print(f"   ↳ Reason: Probability ({prob_increase*100:.1f}%) sits in neutral territory (between 40% and 60%).")
print("="*45)
