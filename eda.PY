import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
print("Loading Credit Card Fraud Dataset...")
df = pd.read_csv('creditcard.csv')
print("Dataset Shape:", df.shape)
print("Fraud Rate:", df['Class'].value_counts(normalize=True)[1])

# Compact EDA - Fits perfectly on screen
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)  # Perfect screen size
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
fig.suptitle('Credit Card Fraud Detection - EDA', fontsize=14, fontweight='bold', y=0.98)

# 1. Class Distribution
class_counts = df['Class'].value_counts()
axes[0,0].pie(class_counts.values, labels=['Normal', 'Fraud'], autopct='%1.1f%%', 
              colors=['#2ecc71', '#e74c3c'], textprops={'fontsize': 10})
axes[0,0].set_title('Class Distribution', fontsize=11, fontweight='bold')

# 2. Amount Distribution
axes[0,1].hist(df[df['Class']==0]['Amount'], bins=30, alpha=0.7, label='Normal', color='#3498db')
axes[0,1].hist(df[df['Class']==1]['Amount'], bins=20, alpha=0.9, label='Fraud', color='#e74c3c')
axes[0,1].set_title('Amount by Class', fontsize=11, fontweight='bold')
axes[0,1].legend(fontsize=9)
axes[0,1].tick_params(labelsize=9)

# 3. Time Pattern
df['Hour'] = pd.to_datetime(df['Time'], unit='s').dt.hour
sns.histplot(data=df, x='Hour', hue='Class', multiple='stack', bins=24, ax=axes[0,2])
axes[0,2].set_title('Transactions by Hour', fontsize=11, fontweight='bold')
axes[0,2].tick_params(labelsize=9)

# 4. Top Fraud Correlations (Compact)
fraud_corr = df.corr()['Class'].sort_values(ascending=False)[1:8]
axes[1,0].barh(range(len(fraud_corr)), fraud_corr.values, color='#e74c3c')
axes[1,0].set_yticks(range(len(fraud_corr)))
axes[1,0].set_yticklabels([x[:4] for x in fraud_corr.index], fontsize=9)
axes[1,0].set_title('Top Fraud Features', fontsize=11, fontweight='bold')
axes[1,0].tick_params(labelsize=9)

# 5. V14 vs V12 (Smaller scatter)
scatter = axes[1,1].scatter(df['V14'], df['V12'], c=df['Class'], cmap='coolwarm', 
                           alpha=0.6, s=8)  # Smaller dots
axes[1,1].set_title('V14 vs V12 Pattern', fontsize=11, fontweight='bold')
axes[1,1].tick_params(labelsize=9)

# 6. Amount Boxplot
sns.boxplot(data=df, x='Class', y='Amount', ax=axes[1,2], palette=['#3498db', '#e74c3c'])
axes[1,2].set_title('Amount Boxplot', fontsize=11, fontweight='bold')
axes[1,2].tick_params(labelsize=9)

plt.tight_layout()
plt.subplots_adjust(top=0.93)  # Perfect margins
plt.savefig('eda_results.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Compact insights
print("\n" + "="*50)
print("EDA SUMMARY:")
print("="*50)
print(f"• {df.shape[0]:,} transactions | {df['Class'].sum():,} frauds ({df['Class'].mean()*100:.2f}%)")
print(f"• Top features: V14, V12, V17")
print(f"• Fraud avg amount: ${df[df['Class']==1]['Amount'].mean():.0f}")
print(f"• Normal avg amount: ${df[df['Class']==0]['Amount'].mean():.0f}")
print("\n✅ COMPACT EDA COMPLETE! Screenshot ready.")
print("="*50)
