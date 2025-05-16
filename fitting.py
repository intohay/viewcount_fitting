import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, fsolve
import matplotlib.pyplot as plt
import japanize_matplotlib

# 1. データ読み込み
df = pd.read_csv('love_yourself_views.csv')

df['datetime'] = pd.to_datetime(df['datetime'])

df['t_num'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds() / 86400

# t: 時間（例: 日数や経過時間）, S: 視聴回数
t = df['t_num'].values
S = df['views'].values

# 2. フィッティング関数定義
def model(t, M, lambd, k):
    S0 = S[0]  # S(0)はデータの最初の値
    return S0 + (M - S0) * (1 - np.exp(-lambd * t)) + k * t

# 3. 初期値の推定
M_init = S.max()
lambd_init = 0.01
k_init = 0.0
p0 = [M_init, lambd_init, k_init]

# 4. フィッティング
params, cov = curve_fit(model, t, S, p0=p0)
M_fit, lambd_fit, k_fit = params

print(f"estimated parameters: M={M_fit}, λ={lambd_fit}, k={k_fit}")

# 5. フィット結果の描画
plt.figure(figsize=(10, 6))  # グラフサイズを大きく

plt.scatter(t, S, label='実データ', color='#1f77b4', edgecolor='black', s=15, alpha=0.8, marker='o')
plt.plot(t, model(t, *params), label='フィッティング', color='#d62728', linewidth=2.5)

plt.title('Love yourself! 視聴回数のフィッティング', fontsize=18, fontweight='bold')
plt.xlabel('経過日数', fontsize=14)
plt.ylabel('視聴回数', fontsize=14)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='best', frameon=True, fancybox=True, framealpha=0.8, borderpad=1)

# パラメータ注釈
param_text = f"M={M_fit:.0f}\nλ={lambd_fit:.4f}\nk={k_fit:.2f}"
plt.gca().text(0.05, 0.95, param_text, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# 100万回に達するt（日数）を求める
target_views = 1_000_000
def reach_target(t):
    return model(t, M_fit, lambd_fit, k_fit) - target_views

t_1m = fsolve(reach_target, x0=[t[-1]])[0]  # 現在の最大日数から外挿

# 到達日時を計算
dt_1m = df['datetime'].iloc[0] + pd.Timedelta(days=t_1m)

print(f"100万回に達する推定日数: {t_1m:.2f} 日後")
print(f"100万回に達する推定日時: {dt_1m}")

# フィッティング曲線を100万回到達時点まで拡張
t_fit = np.linspace(t[0], t_1m, 300)
S_fit = model(t_fit, M_fit, lambd_fit, k_fit)

plt.figure(figsize=(10, 6))
plt.scatter(t, S, label='実データ', color='#1f77b4', edgecolor='black', s=15, alpha=0.8, marker='o')
plt.plot(t_fit, S_fit, label='フィッティング', color='#d62728', linewidth=1.5)

plt.title('Love yourself! 視聴回数のフィッティング', fontsize=18, fontweight='bold')
plt.xlabel('経過日数', fontsize=14)
plt.ylabel('視聴回数', fontsize=14)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='best', frameon=True, fancybox=True, framealpha=0.8, borderpad=1)

# パラメータ注釈
param_text = f"M={M_fit:.0f}\nλ={lambd_fit:.4f}\nk={k_fit:.2f}"
plt.gca().text(0.05, 0.95, param_text, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# 100万回到達予測の縦線と注釈（時刻まで表示）
plt.axvline(t_1m, color='green', linestyle=':', linewidth=2, label='100万回到達予測')
plt.gca().text(
    t_1m, target_views,
    f"{dt_1m.strftime('%Y-%m-%d %H:%M')}\n100万回",
    color='green', fontsize=12, verticalalignment='bottom', horizontalalignment='left'
)

plt.tight_layout()
plt.show()