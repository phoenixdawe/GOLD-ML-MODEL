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

# Secure S&P 500 download
sp500 = yf.download("^GSPC", start=START_DATE, end=END_DATE)[["Close"]].rename(columns={"Close": "^GSPC"})

# Robust 10-Year yield download with structural try-except safety fallback
try:
    print("📈 Fetching US 10-Year Treasury Yield Index (^TNX)...")
    tnx = yf.download("^TNX", start=START_DATE, end=END_DATE)[["Close"]].rename(columns={"Close": "^TNX"})
    if tnx.dropna().empty:
        raise ValueError("YFinance returned an empty matrix for ^TNX.")
except Exception as e:
    print(f"⚠️ YFinance ^TNX download failed ({e}). Pulling stable FRED DGS10 asset instead...")
    # Alternate path: Pulling official daily constant maturity 10Y yields directly from FRED
    dgs10_raw = fred.get_series("DGS10", start=START_DATE, end=END_DATE)
    tnx = pd.DataFrame({"^TNX": dgs10_raw})

# Secure Gold Futures download
gold = yf.download("GC=F", start=START_DATE, end=END_DATE)[["Close"]].rename(columns={"Close": "GC=F"})

# Standardise column layers to smash MultiIndex bugs completely
for df_temp in [sp500, tnx, gold]:
    df_temp.columns = df_temp.columns.get_level_values(0) if isinstance(df_temp.columns, pd.MultiIndex) else df_temp.columns

# Extract FRED variables
fred_series = {"expected_inf_1y": "MICH", "expected_inf_5y": "T5YIE", "real_interest_rate": "DFII10"}
fred_dfs = []
for name, series_id in fred_series.items():
    s_data = fred.get_series(series_id, start=START_DATE, end=END_DATE)
    fred_dfs.append(pd.DataFrame({name: s_data}))

# Align timelines explicitly using sort=False to prevent warning notices
master_df = pd.concat([sp500, tnx, gold] + fred_dfs, axis=1, sort=False)

# Forward-fill gaps (e.g. macro stats released on weekends or delays), then trim any trailing NaN edges
master_df = master_df.ffill().dropna()

# Compute Structural Technical Features
master_df["sp500_roc"] = master_df["^GSPC"].pct_change(1)
master_df["us_10y_roc"] = master_df["^TNX"].pct_change(1)

rolling_mean = master_df["GC=F"].rolling(window=20).mean()
rolling_std = master_df["GC=F"].rolling(window=20).std()
master_df["gold_zscore"] = (master_df["GC=F"] - rolling_mean) / rolling_std

# ==========================================
# 2. LEAK-PROOF TARGETS & VOLATILITY CONTROLS
# ==========================================
print("🔧 Applying leak-proof data and risk controls...")

FIXED_CONTRACTS = 2  
COST_PER_CONTRACT = 15.00  # Slippage + execution broker baseline toll

# Volatility Based Risk (14-day rolling ATR estimate)
high_low = master_df["GC=F"] * 0.012  
master_df["atr"] = high_low.rolling(window=14).mean().ffill()

# Leak-Proof Targets
master_df["target_direction"] = np.where(master_df["GC=F"].shift(-2) > master_df["GC=F"], 1, 0)
master_df.dropna(inplace=True)

# Define feature columns
feature_cols = ["sp500_roc", "us_10y_roc", "gold_zscore", "expected_inf_1y", "expected_inf_5y", "real_interest_rate"]

# Split chronological datasets: Train (2023-2025) vs Test (2026 Out-of-Sample)
train_df = master_df[master_df.index < "2026-01-01"].copy()
test_df = master_df[master_df.index >= "2026-01-01"].copy()

if test_df.empty:
    raise ValueError(f"Testing dataset is empty. Data frame shape info: {master_df.shape}. Max date available: {master_df.index.max()}")

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
# 4. LONG-ONLY BACKTEST SIMULATION LOOP WITH FRICTION
# ==========================================
print("⚡ Running friction-adjusted long-only simulation on 2026 data...")
capital = INITIAL_CAPITAL
capital_history = []
long_returns, daily_returns = [], []
sp500_initial = test_df["^GSPC"].iloc[0]

for i in range(len(test_df) - 2):
    current_price = float(test_df["GC=F"].iloc[i])
    future_2d_price = float(test_df["GC=F"].iloc[i+2])
    prob_long = float(probs[i])
    current_atr = float(test_df["atr"].iloc[i])
    
    trade_return = 0.0
    stop_distance = current_atr * 1.5 
    
    if prob_long >= 0.60:  # LONG ONLY POSITION CONDITION
        stop_loss_price = current_price - stop_distance
        
        if float(test_df["GC=F"].iloc[i+1]) <= stop_loss_price or future_2d_price <= stop_loss_price:
            gross_pnl = (-stop_distance) * GOLD_MULTIPLIER * FIXED_CONTRACTS
        else:
            gross_pnl = (future_2d_price - current_price) * GOLD_MULTIPLIER * FIXED_CONTRACTS
            
        friction_penalty = COST_PER_CONTRACT * FIXED_CONTRACTS
        trade_return = gross_pnl - friction_penalty
        long_returns.append(trade_return)
        
    prev_capital = capital
    capital += trade_return
    capital_history.append(capital)
    daily_returns.append((capital - prev_capital) / prev_capital if prev_capital > 0 else 0)

# ==========================================
# 5. RISK METRICS & PERFORMANCE DIAGNOSTICS
# ==========================================
print("🔬 Running deep forensic model diagnostics...")
long_pnl = np.array(long_returns)

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

final_sp500_return = ((test_df["^GSPC"].iloc[-1] - sp500_initial) / sp500_initial) * 100
final_strategy_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
daily_ret_series = pd.Series(daily_returns).dropna()
sharpe_ratio = (daily_ret_series.mean() / daily_ret_series.std()) * np.sqrt(252) if daily_ret_series.std() != 0 else 0.0

print("\n" + "="*60)
print("         🤖 FRICTION-ADJUSTED LONG-ONLY REPORT           ")
print("="*60)
print(f"Total Strategy Return    : {final_strategy_return:+.2f}%  | S&P 500: {final_sp500_return:+.2f}%")
print(f"Portfolio Sharpe Ratio   : {sharpe_ratio:.2f}")
print(f"Total Trades Executed    : {len(long_pnl)}")
print(f"Strategy Win Rate        : {l_win_rate:.1f}%")
print(f"Strategy Profit Factor   : {l_profit_factor:.2f}")
print(f"Avg Win vs Avg Loss      : ${l_avg_win:+,.2f} / ${l_avg_loss:,.2f}")
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
plt.plot(plot_dates, capital_history, label="Friction-Adjusted Long Gold Strategy", color="#1f77b4", linewidth=2.5)
plt.plot(plot_dates, sp500_plot_series, label="S&P 500 Benchmark Index", color="#d62728", linewidth=1.5, linestyle="--")

metrics_textbox = (
    f"Strategy Return: {final_strategy_return:+.1f}%\n"
    f"Win Rate: {l_win_rate:.1f}%\n"
    f"Profit Factor: {l_profit_factor:.2f}"
)
plt.text(0.02, 0.95, metrics_textbox, transform=ax.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))

plt.title("2026 Out-of-Sample Performance (Long-Only with Slippage/Commissions)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Execution Timeline", fontsize=11, labelpad=10)
plt.ylabel("Portfolio Valuation ($ USD)", fontsize=11, labelpad=10)
plt.grid(visible=True, color="#e5e5e5", linestyle="-", linewidth=0.7)
plt.legend(loc="upper right", frameon=True, fontsize=11)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()

# ==========================================
# 7. MONTE CARLO PERMUTATION STRESS TEST
# ==========================================
print("\n🎲 Running 500-Iteration Permuted Monte Carlo Stress Test...")

num_permutations = 500
permuted_returns = []
observed_profit = final_strategy_return

# Extract a clean, localized copy of target variables for speed
prices_1d = test_df["GC=F"].shift(-1).iloc[:len(probs)-2].values
prices_2d = test_df["GC=F"].shift(-2).iloc[:len(probs)-2].values
current_prices = test_df["GC=F"].iloc[:len(probs)-2].values
atrs = test_df["atr"].iloc[:len(probs)-2].values

for sim in range(num_permutations):
    # Randomly shuffle the predictive probability array to destroy the edge
    shuffled_probs = np.random.permutation(probs)
    
    sim_capital = INITIAL_CAPITAL
    
    for i in range(len(shuffled_probs) - 2):
        prob_long = float(shuffled_probs[i])
        current_price = float(current_prices[i])
        future_2d_price = float(prices_2d[i])
        current_atr = float(atrs[i])
        
        trade_return = 0.0
        stop_distance = current_atr * 1.5
        
        if prob_long >= 0.60:  # LONG ONLY POSITION CONDITION
            stop_loss_price = current_price - stop_distance
            if float(prices_1d[i]) <= stop_loss_price or future_2d_price <= stop_loss_price:
                gross_pnl = (-stop_distance) * GOLD_MULTIPLIER * FIXED_CONTRACTS
            else:
                gross_pnl = (future_2d_price - current_price) * GOLD_MULTIPLIER * FIXED_CONTRACTS
                
            trade_return = gross_pnl - (COST_PER_CONTRACT * FIXED_CONTRACTS)
            
        sim_capital += trade_return
        
    sim_return = ((sim_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    permuted_returns.append(sim_return)

# Compute the Empirical P-Value
permuted_returns = np.array(permuted_returns)
beaten_by_noise = np.sum(permuted_returns >= observed_profit)
p_value = beaten_by_noise / num_permutations

print("\n" + "="*60)
print("          🎲 MONTE CARLO PERMUTATION RESULTS            ")
print("="*60)
print(f"Observed Strategy Return       : {observed_profit:+.2f}%")
print(f"Permuted Median Random Return  : {np.median(permuted_returns):+.2f}%")
print(f"Permuted Maximum Random Return  : {np.max(permuted_returns):+.2f}%")
print(f"Times Random Noise Beat Strategy: {beaten_by_noise} / {num_permutations}")
print(f"Calculated Strategy P-Value    : {p_value:.4f}")

if p_value < 0.05:
    print("🟢 STATUS: SIGNIGICANT EDGE. Less than 5% chance this was luck.")
else:
    print("🔴 STATUS: OVERFITTED NOISE. Random entries match your performance.")
print("="*60)

# Optional Plotting histogram of the noise distribution
plt.figure(figsize=(10, 5))
plt.hist(permuted_returns, bins=30, color="#cccccc", edgecolor="#999999", label="Random Signal Noise Distribution")
plt.axvline(observed_profit, color="#1f77b4", linewidth=2.5, linestyle="--", label=f"Your Real Strategy ({observed_profit:.1f}%)")
plt.title("Permuted Monte Carlo Null Distribution vs Real Performance")
plt.xlabel("Total Out-of-Sample Return (%)")
plt.ylabel("Frequency Count")
plt.legend()
plt.show()
