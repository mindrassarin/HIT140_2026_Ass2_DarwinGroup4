#HIT140 FOUNDATIONS OF DATA SCIENCE
#ASSIGNMENT 2
# STUDENT ID : S404210
#NAME:BETTY JELAGAT


import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import os



# 1. LOAD DATA WITH CORRECT ENCODING
csv_path = 'Player_Shots_on_Target_data.csv'
if not os.path.exists(csv_path):
    csv_path = '/workspace/knowledge/Player_Shots_on_Target_data.csv'

df = pd.read_csv(csv_path, encoding='cp1252')
# 2. DATA WRANGLING
# Select variables relevant to positional shooting analysis
df_wrangle = df[['Player', 'Pos', 'Squad', 'Age', '90s', 'Sh', 'SoT', 'SoT/90']].copy()

# Filter out players who never played in any match (non-participants)
df_wrangle = df_wrangle[df_wrangle['90s'] > 0].copy()

# Extract first listed position as Primary Position
df_wrangle['Primary_Position'] = df_wrangle['Pos'].str.split(',').str[0]

# Filter for Forwards (FW) and Midfielders (MF) only
df_wrangle = df_wrangle[df_wrangle['Primary_Position'].isin(['FW', 'MF'])].copy()

# Map abbreviations to clearer descriptive position labels
df_wrangle['Position_Group'] = df_wrangle['Primary_Position'].map({'FW': 'Forward', 'MF': 'Midfielder'})

# Drop any missing values in our target variable or position group
df_clean = df_wrangle.dropna(subset=['SoT/90', 'Position_Group']).copy()

# ASSIGN UNIQUE PLAYER ID TO CLEAN DATA FOR REPRODUCIBLE SAMPLING
df_clean['Player_ID'] = ['PL' + str(i).zfill(4) for i in range(1, len(df_clean) + 1)]

# 3. POPULATION PARAMETERS FOR COMPARISON & ACCURACY CHECK
pop_fws = df_clean[df_clean['Position_Group'] == 'Forward']['SoT/90']
pop_mids = df_clean[df_clean['Position_Group'] == 'Midfielder']['SoT/90']

print("=== POPULATION PARAMETERS ===")
print(f"Population Size - Forwards: {len(pop_fws)}, Midfielders: {len(pop_mids)}")
print(f"True Population Mean (Forwards): {pop_fws.mean():.4f}")
print(f"True Population Mean (Midfielders): {pop_mids.mean():.4f}")
print("=============================\n")

# 4. REPRODUCIBLE STRATIFIED RANDOM SAMPLING USING PLAYER IDs
np.random.seed(42) # Set seed to ensure identical sample results across runs

# Get the lists of player IDs for each position group
fw_ids = df_clean[df_clean['Position_Group'] == 'Forward']['Player_ID']
mid_ids = df_clean[df_clean['Position_Group'] == 'Midfielder']['Player_ID']

# Randomly select 64 Player IDs from each group
sample_fw_ids = fw_ids.sample(n=64, random_state=42)
sample_mid_ids = mid_ids.sample(n=64, random_state=42)

# Extract SoT/90 values associated with the sampled Player IDs
sample_fw = df_clean[df_clean['Player_ID'].isin(sample_fw_ids)]['SoT/90']
sample_mid = df_clean[df_clean['Player_ID'].isin(sample_mid_ids)]['SoT/90']

# 5. DESCRIPTIVE STATISTICS FOR SAMPLES
print("=== DESCRIPTIVE STATISTICS FOR SAMPLES ===")
print("\n--- Forwards Sample Stats ---")
print(sample_fw.describe())
print("\n--- Midfielders Sample Stats ---")
print(sample_mid.describe())
print("==========================================\n")

# 6. CHECK NORMALITY & GENERATE PLOT (Interactive with plt.show())
print("=== NORMALITY ASSUMPTION CHECK ===")
fw_shapiro_p = st.shapiro(sample_fw)[1]
mid_shapiro_p = st.shapiro(sample_mid)[1]
print(f"Forwards Shapiro-Wilk p-value: {fw_shapiro_p:.4f}")
print(f"Midfielders Shapiro-Wilk p-value: {mid_shapiro_p:.4f}")

plt.figure(figsize=(12, 5))

# Subplot 1: Forwards Sample Distribution
plt.subplot(1, 2, 1)
plt.hist(sample_fw, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Sample Forwards SoT/90')
plt.xlabel('Shots on Target per 90 (SoT/90)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 2: Midfielders Sample Distribution
plt.subplot(1, 2, 2)
plt.hist(sample_mid, bins=10, color='salmon', edgecolor='black', alpha=0.7)
plt.title('Sample Midfielders SoT/90')
plt.xlabel('Shots on Target per 90 (SoT/90)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# Save a copy as an image file first (for assignments/reports)
plt.savefig('shooting_precision_histogram.png', dpi=150)

# Render the plot in your IDE's interactive window (safely caught for headless execution)
try:
    plt.show()
except Exception:
    print("No graphical display detected. The plot window could not be opened, but the histogram was successfully saved to disk as 'shooting_precision_histogram.png'.")
print("==================================\n")

# 7. CONFIDENCE INTERVALS (95% CI)
fw_mean, fw_std, n_fw = sample_fw.mean(), sample_fw.std(), len(sample_fw)
mid_mean, mid_std, n_mid = sample_mid.mean(), sample_mid.std(), len(sample_mid)

# 95% Confidence Level corresponds to z* = 1.960
z_critical = 1.960

fw_se = fw_std / np.sqrt(n_fw)
fw_ci = (fw_mean - z_critical * fw_se, fw_mean + z_critical * fw_se)

mid_se = mid_std / np.sqrt(n_mid)
mid_ci = (mid_mean - z_critical * mid_se, mid_mean + z_critical * mid_se)

print("=== 95% CONFIDENCE INTERVALS OF THE MEAN ===")
print(f"Forwards 95% CI: ({fw_ci[0]:.4f}, {fw_ci[1]:.4f}) -> True Population Mean ({pop_fws.mean():.4f}) is Enclosed: {fw_ci[0] <= pop_fws.mean() <= fw_ci[1]}")
print(f"Midfielders 95% CI: ({mid_ci[0]:.4f}, {mid_ci[1]:.4f}) -> True Population Mean ({pop_mids.mean():.4f}) is Enclosed: {mid_ci[0] <= pop_mids.mean() <= mid_ci[1]}")
print("============================================\n")

# 8. INFERENTIAL HYPOTHESIS TESTING (Welch Two-Sample t-Test)
# null hypothesis: mean of sample 1 = mean of sample 2
# alternative hypothesis: mean of sample 1 is greater than mean of sample 2 (one-sided test)
# equal_var=False assumes that two populations do not have equal variance
t_stat, p_val_greater = st.ttest_ind(sample_fw, sample_mid, equal_var=False, alternative='greater')

print("=== WELCH TWO-SAMPLE T-TEST RESULTS ===")
print(f"Welch t-statistic: {t_stat:.4f}")
print(f"One-sided p-value (greater): {p_val_greater:.4e}")

alpha = 0.05
print("\nConclusion:")
if p_val_greater <= alpha:
    print(f"\tWe reject the null hypothesis at alpha = {alpha}.")
    print("\tThere is sufficient evidence to conclude that Forwards record a significantly higher average SoT/90 than Midfielders.")
else:
    print(f"\tWe fail to reject the null hypothesis at alpha = {alpha}.")
    print("\tThere is insufficient evidence to conclude that Forwards record a significantly higher average SoT/90 than Midfielders.")
print("=======================================")
