import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

print("FRAUD DETECTION - TRAINING 4 ML MODELS")
print("="*60)

# Load and preprocess data
df = pd.read_csv('creditcard.csv')
print(f"Dataset: {df.shape[0]:,} transactions | Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")

# Preprocessing
scaler = RobustScaler()
X = scaler.fit_transform(df.drop('Class', axis=1))
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE for imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE: {X_train_smote.shape} | Balanced: {np.bincount(y_train_smote)}")

print("\nTRAINING MODELS...")
print("-" * 40)

# Define models
models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'),
    'IsolationForest': IsolationForest(contamination=0.0017, random_state=42)
}

results = {}
predictions = {}
probas = {}

for name, model in models.items():
    print(f"Training {name}...", end=" ")
    
    try:
        # Train models
        if name != 'IsolationForest':
            model.fit(X_train_smote, y_train_smote)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train)
            y_pred = model.predict(X_test)
            y_proba = model.decision_function(X_test)
        
        # Metrics
        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['1']['precision']
        recall = report['1']['recall']
        
        results[name] = {'AUC': auc, 'Precision': precision, 'Recall': recall}
        predictions[name] = y_pred
        probas[name] = y_proba
        
        print(f"AUC: {auc:.4f}")
        
        # Save XGBoost separately
        if 'XGBoost' in name:
            joblib.dump(model, 'xgboost_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            print("xgboost_model.pkl SAVED!")
            
    except Exception as e:
        print(f"Error: {e}")
        continue

print("\nRESULTS TABLE:")
print("-" * 40)
results_df = pd.DataFrame(results).T.round(4)
print(results_df)
results_df.to_csv('model_results.csv')
print("model_results.csv SAVED!")

# Find best model
best_model_name = max(results.keys(), key=lambda k: results[k]['AUC'])
print(f"\nBEST MODEL: {best_model_name} (AUC: {results[best_model_name]['AUC']:.4f})")

# VISUALIZATIONS
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Fraud Detection Model Comparison', fontsize=14, fontweight='bold')

# 1. AUC Bar Chart
auc_scores = results_df['AUC']
colors = ['gold' if x == results_df['AUC'].max() else 'skyblue' for x in auc_scores]
axes[0].bar(auc_scores.index, auc_scores.values, color=colors, edgecolor='black')
axes[0].set_title('AUC Scores (Higher = Better)')
axes[0].set_ylabel('AUC Score')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(auc_scores.values):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')

# 2. Precision vs Recall Scatter
axes[1].scatter(results_df['Precision'], results_df['Recall'], s=200, alpha=0.8)
for name in results_df.index:
    axes[1].annotate(name[:4], (results_df.loc[name, 'Precision'], results_df.loc[name, 'Recall']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
axes[1].set_xlabel('Precision')
axes[1].set_ylabel('Recall')
axes[1].set_title('Precision vs Recall Trade-off')
axes[1].grid(True, alpha=0.3)

# 3. Confusion Matrix (Best Model)
cm = confusion_matrix(y_test, predictions[best_model_name])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
axes[2].set_title(f'Confusion Matrix\n{best_model_name}')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("model_comparison.png SAVED!")

# Feature importance (XGBoost)
if 'XGBoost' in models:
    xgb_model = models['XGBoost']
    feature_names = df.drop('Class', axis=1).columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Top 10 Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("feature_importance.png SAVED!")
    importance_df.to_csv('feature_importance.csv')

print("\nALL FILES GENERATED SUCCESSFULLY!")
print("="*60)
print("FILES CREATED:")
print("• model_results.csv")
print("• model_comparison.png") 
print("• xgboost_model.pkl")
print("• scaler.pkl")
print("• feature_importance.png")
print("• feature_importance.csv")
print("\nPROJECT READY FOR DASHBOARD!")
