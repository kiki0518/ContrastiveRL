import pandas as pd
import matplotlib.pyplot as plt

# 1. 讀取資料
df_log = pd.read_csv("training_log.csv")

# 2. 定義間隔 (每 500 step 一組)
bin_size = 500
df_log['step_bin'] = (df_log['step'] // bin_size) * bin_size

# 3. 計算各組的平均值與標準差
# 這裡以 L_RM 為例，你可以換成 L_total
stats = df_log.groupby(['exp', 'step_bin'])['L_RM'].agg(['mean', 'std']).reset_index()

# 4. 開始繪圖
plt.figure(figsize=(10, 6))

for exp in stats['exp'].unique():
    subset = stats[stats['exp'] == exp]
    
    # 畫出平均值線
    line = plt.plot(subset['step_bin'], subset['mean'], marker='o', label=f'{exp} (Mean)')
    
    # 畫出標準差陰影 (fill_between)
    # 如果某組只有一筆資料，std 會是 NaN，可以用 fillna(0) 處理
    plt.fill_between(
        subset['step_bin'], 
        subset['mean'] - subset['std'].fillna(0), 
        subset['mean'] + subset['std'].fillna(0), 
        alpha=0.2,
        color=line[0].get_color() # 讓陰影顏色與線條一致
    )

plt.xlabel(f"Training Step (Binned by {bin_size})")
plt.ylabel("Loss Value (L_RM)")
plt.title("Reward Model Training: E3 vs E4 (Mean ± Std)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 儲存圖片
plt.savefig("E3_E4_Comparison.png")
plt.show()