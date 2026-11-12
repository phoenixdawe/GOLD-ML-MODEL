import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
FRED_API_KEY = "601b81ed717c38ac7dd912053b08c754"
fred = Fred(api_key=FRED_API_KEY)

START_DATE = "2023-01-01"
END_DATE = "2026-07-01" 
INITIAL_CAPITAL = 100000.0
GOLD_MULTIPLIER = 100  # 1 GC=F Contract = 100 ounces

print("🤖 Downloading structured financial datasets...")
sp500 = yf.download("^GSPC", start=START_DATE, end=END_DATE)[["Close"]].rename(columns={"Close": "^GSPC"})
tnx = yf.download("^TNX", start=START_DATE, end=END_DATE)[["Close"]].rename(columns={"Close": "^TNX"})
gold = yf.download("GC=F", start=START_DATE, end=END_DATE)[["Close"]].rename(columns={"Close": "GC=F"})

# Remove MultiIndex headers from yfinance output layers
for df_temp in [sp500, tnx, gold]:
    df_temp.columns = df_temp.columns.get_level_values(0) if isinstance(df_temp.columns, pd.MultiIndex) else df_temp.columns

# Extract FRED variables
fred_series = {"expected_inf_1y": "MICH", "expected_inf_5y": "T5YIE", "real_interest_rate": "DFII10"}
fred_dfs = []
for name, series_id in fred_series.items():
    s_data = fred.get_series(series_id, start=START_DATE, end=END_DATE)
    fred_dfs.append(pd.DataFrame({name: s_data}))

# Align timelines (added sort=False to resolve the Pandas4Warning)
master_df = pd.concat([sp500, tnx, gold] + fred_dfs, axis=1, sort=False).ffill().dropna()

# Compute Features
master_df["sp500_roc"] = master_df["^GSPC"].pct_change(1)
master_df["us_10y_roc"] = master_df["^TNX"].pct_change(1)

rolling_mean = master_df["GC=F"].rolling(window=20).mean()
rolling_std = master_df["GC=F"].rolling(window=20).std()
master_df["gold_zscore"] = (master_df["GC=F"] - rolling_mean) / rolling_std

# ==========================================
# 2. LEAK-PROOF TARGETS & VOLATILITY CONTROLS
# ==========================================
print("🔧 Applying leak-proof data and risk controls...")

# 1. FIXED CONTRACT SIZE 
FIXED_CONTRACTS = 2  

# 2. VOLATILITY BASED RISK (14-day rolling ATR estimate)
high_low = master_df["GC=F"] * 0.012  
master_df["atr"] = high_low.rolling(window=14).mean().ffill()

# 3. LEAK-PROOF TARGETS
master_df["target_direction"] = np.where(master_df["GC=F"].shift(-2) > master_df["GC=F"], 1, 0)
master_df.dropna(inplace=True)

# Define feature columns
feature_cols = ["sp500_roc", "us_10y_roc", "gold_zscore", "expected_inf_1y", "expected_inf_5y", "real_interest_rate"]

# Split chronological datasets: Train (2023-2025) vs Test (2026 Out-of-Sample)
train_df = master_df[master_df.index < "2026-01-01"].copy()
test_df = master_df[master_df.index >= "2026-01-01"].copy()

if test_df.empty:
    raise ValueError("Testing dataset is empty. Ensure END_DATE extends safely into 2026.")

# FIXED: Scaler initialization added here
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])

# ==========================================
# 3. MODEL TRAINING (CLASSIFICATION ONLY)
# ==========================================
print("🧠 Training ML Pipeline on 2023-2025 regimes...")
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, train_df["target_direction"])
probs = clf.predict_proba(X_test)[:, 1]

# ==========================================
# 4. BACKTEST SIMULATION LOOP
# ==========================================
print("⚡ Running leak-free out-of-sample simulation on 2026 data...")
capital = INITIAL_CAPITAL
capital_history = []
long_returns, short_returns, daily_returns = [], [], []
sp500_initial = test_df["^GSPC"].iloc[0]

for i in range(len(test_df) - 2):
    current_price = float(test_df["GC=F"].iloc[i])
    future_2d_price = float(test_df["GC=F"].iloc[i+2])
    prob_long = float(probs[i])
    current_atr = float(test_df["atr"].iloc[i])
    
    trade_return = 0.0
    stop_distance = current_atr * 1.5 
    
    if prob_long >= 0.60:  # LONG
        stop_loss_price = current_price - stop_distance
        if float(test_df["GC=F"].iloc[i+1]) <= stop_loss_price or future_2d_price <= stop_loss_price:
            trade_return = (-stop_distance) * GOLD_MULTIPLIER * FIXED_CONTRACTS
        else:
            trade_return = (future_2d_price - current_price) * GOLD_MULTIPLIER * FIXED_CONTRACTS
        long_returns.append(trade_return)
        
    elif prob_long <= 0.40:  # SHORT
        stop_loss_price = current_price + stop_distance
        if float(test_df["GC=F"].iloc[i+1]) >= stop_loss_price or future_2d_price >= stop_loss_price:
            trade_return = (-stop_distance) * GOLD_MULTIPLIER * FIXED_CONTRACTS
        else:
            trade_return = (current_price - future_2d_price) * GOLD_MULTIPLIER * FIXED_CONTRACTS
        short_returns.append(trade_return)
        
    prev_capital = capital
    capital += trade_return
    capital_history.append(capital)
    daily_returns.append((capital - prev_capital) / prev_capital if prev_capital > 0 else 0)

# ==========================================
# 5. RISK METRICS & PERFORMANCE DIAGNOSTICS
# ==========================================
print("🔬 Running deep forensic model diagnostics...")
long_pnl = np.array(long_returns)
short_pnl = np.array(short_returns)
all_pnl = np.concatenate([long_pnl, short_pnl]) if len(long_returns) + len(short_returns) > 0 else np.array([])

def calculate_trade_metrics(pnl_array):
    if len(pnl_array) == 0: return 0.0, 0.0, 0.0, 0.0
    wins = pnl_array[pnl_array > 0]
    losses = pnl_array[pnl_array <= 0]
    win_rate = (len(wins) / len(pnl_array)) * 100
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 else np.inf
    return win_rate, avg_win, avg_loss, profit_factor

l_win_rate, l_avg_win, l_avg_loss, l_profit_factor = calculate_trade_metrics(long_pnl)
s_win_rate, s_avg_win, s_avg_loss, s_profit_factor = calculate_trade_metrics(short_pnl)
tot_win_rate, tot_avg_win, tot_avg_loss, tot_profit_factor = calculate_trade_metrics(all_pnl)

final_sp500_return = ((test_df["^GSPC"].iloc[-1] - sp500_initial) / sp500_initial) * 100
final_strategy_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
daily_ret_series = pd.Series(daily_returns).dropna()
sharpe_ratio = (daily_ret_series.mean() / daily_ret_series.std()) * np.sqrt(252) if daily_ret_series.std() != 0 else 0.0

print("\n" + "="*60)
print("             🤖 CLEAN MODEL PERFORMANCE REPORT           ")
print("="*60)
print(f"Total Strategy Return    : {final_strategy_return:+.2f}%  | S&P 500: {final_sp500_return:+.2f}%")
print(f"Portfolio Sharpe Ratio   : {sharpe_ratio:.2f}")
print(f"Overall Strategy Win Rate: {tot_win_rate:.1f}% ({len(all_pnl)} total trades)")
print(f"Overall Profit Factor    : {tot_profit_factor:.2f}")
print("\n📈 LONG TRADE METRICS:")
print(f"  ↳ Long Win Rate        : {l_win_rate:.1f}% ({len(long_pnl)} trades)")
print(f"  ↳ Avg Win vs Avg Loss  : ${l_avg_win:+,.2f} / ${l_avg_loss:,.2f}")
print(f"  ↳ Long Profit Factor   : {l_profit_factor:.2f}")
print("\n📉 SHORT TRADE METRICS:")
print(f"  ↳ Short Win Rate       : {s_win_rate:.1f}% ({len(short_pnl)} trades)")
print(f"  ↳ Avg Win vs Avg Loss  : ${s_avg_win:+,.2f} / ${s_avg_loss:,.2f}")
print(f"  ↳ Short Profit Factor  : {s_profit_factor:.2f}")
print("="*60)

# ==========================================
# 6. VISUALISATION ENGINE (MATPLOTLIB)
# ==========================================
import matplotlib.pyplot as plt
plot_dates = test_df.index[:len(capital_history)]
sp500_normalized = (test_df["^GSPC"] / sp500_initial) * INITIAL_CAPITAL
sp500_plot_series = sp500_normalized.iloc[:len(capital_history)]

plt.figure(figsize=(15, 7), facecolor="#f8f9fa")
ax = plt.axes()
ax.set_facecolor("#ffffff")
plt.plot(plot_dates, capital_history, label="Leveraged ML Gold Strategy", color="#1f77b4", linewidth=2.5)
plt.plot(plot_dates, sp500_plot_series, label="S&P 500 Benchmark Index", color="#d62728", linewidth=1.5, linestyle="--")

metrics_textbox = (
    f"Strategy Return: {final_strategy_return:+.1f}%\n"
    f"Overall Win Rate: {tot_win_rate:.1f}%\n"
    f"Profit Factor: {tot_profit_factor:.2f}"
)
plt.text(0.02, 0.95, metrics_textbox, transform=ax.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))

plt.title("2026 Out-of-Sample Performance (Leak-Free Calibration)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Execution Timeline", fontsize=11, labelpad=10)
plt.ylabel("Portfolio Valuation ($ USD)", fontsize=11, labelpad=10)
plt.grid(visible=True, color="#e5e5e5", linestyle="-", linewidth=0.7)
plt.legend(loc="upper right", frameon=True, fontsize=11)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()
