"""
Telecom Customer Churn Prediction
===================================
Business Problem:
    A telecom company loses revenue every time a customer churns.
    Acquiring a new customer costs 5-7x more than retaining an existing one.
    This project builds a model to identify at-risk customers before they leave,
    enabling proactive retention campaigns.

Dataset: IBM Telco Customer Churn (publicly available)
    - 7,043 customers | 21 features
    - Target: Churn (Yes/No)

Approach:
    1. Exploratory Data Analysis (EDA)
    2. Feature Engineering & Preprocessing
    3. Model Training: Logistic Regression + Random Forest
    4. Evaluation: ROC-AUC, Precision-Recall, Confusion Matrix
    5. Business Interpretation: Feature Importance & Risk Segmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)

# ─────────────────────────────────────────────
# 1. GENERATE REALISTIC DATASET
#    (mirrors IBM Telco Churn structure)
# ─────────────────────────────────────────────

np.random.seed(42)
n = 7043

def generate_telco_data(n):
    tenure = np.random.exponential(32, n).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(65, 30, n).clip(18, 120)
    total_charges = (tenure * monthly_charges * np.random.uniform(0.9, 1.1, n)).clip(0)

    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                 n, p=[0.55, 0.24, 0.21])
    internet = np.random.choice(['DSL', 'Fiber optic', 'No'],
                                  n, p=[0.34, 0.44, 0.22])
    payment = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        n, p=[0.34, 0.23, 0.22, 0.21])

    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70])
    phone = np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10])
    multiple_lines = np.where(phone == 'Yes',
                               np.random.choice(['Yes', 'No'], n, p=[0.42, 0.58]),
                               'No phone service')
    online_security = np.where(internet != 'No',
                                np.random.choice(['Yes', 'No'], n, p=[0.29, 0.71]),
                                'No internet service')
    tech_support = np.where(internet != 'No',
                             np.random.choice(['Yes', 'No'], n, p=[0.29, 0.71]),
                             'No internet service')
    paperless = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])

    # churn probability: driven by contract, tenure, monthly charges, fiber
    churn_prob = (
        0.05
        + (contract == 'Month-to-month') * 0.22
        + (internet == 'Fiber optic') * 0.12
        + (payment == 'Electronic check') * 0.08
        + (online_security == 'No') * 0.06
        + (tech_support == 'No') * 0.05
        + (paperless == 'Yes') * 0.04
        + (senior == 1) * 0.05
        - (tenure / 72) * 0.25
        - (partner == 'Yes') * 0.03
        + np.random.normal(0, 0.05, n)
    ).clip(0.02, 0.90)

    churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    return pd.DataFrame({
        'customerID': [f'ID-{i:05d}' for i in range(n)],
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges.round(2),
        'TotalCharges': total_charges.round(2),
        'Churn': churn
    })

df = generate_telco_data(n)
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['Churn'].mean():.1%}")
print("\nSample data:")
print(df.head(3).to_string())


# ─────────────────────────────────────────────
# 2. EDA PLOTS
# ─────────────────────────────────────────────

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'churn': '#E63946', 'retain': '#457B9D', 'accent': '#2B2D42'}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Telecom Churn — Exploratory Data Analysis', fontsize=16,
             fontweight='bold', y=1.01)

# 1. Churn distribution
ax = axes[0, 0]
labels = ['Retained', 'Churned']
sizes = df['Churn'].value_counts().sort_index().values
colors = [COLORS['retain'], COLORS['churn']]
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                   autopct='%1.1f%%', startangle=90,
                                   textprops={'fontsize': 11})
autotexts[1].set_fontweight('bold')
ax.set_title('Overall Churn Rate', fontweight='bold')

# 2. Churn by Contract
ax = axes[0, 1]
ct = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
bars = ax.bar(ct.index, ct.values * 100, color=[COLORS['churn'], '#F4A261', COLORS['retain']])
ax.set_title('Churn Rate by Contract Type', fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, 55)
for bar, val in zip(bars, ct.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
ax.tick_params(axis='x', rotation=10)

# 3. Tenure distribution by churn
ax = axes[0, 2]
ax.hist(df[df['Churn'] == 0]['tenure'], bins=30, alpha=0.6,
        color=COLORS['retain'], label='Retained', density=True)
ax.hist(df[df['Churn'] == 1]['tenure'], bins=30, alpha=0.6,
        color=COLORS['churn'], label='Churned', density=True)
ax.set_title('Tenure Distribution by Churn', fontweight='bold')
ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Density')
ax.legend()
ax.axvline(df[df['Churn']==1]['tenure'].median(), color=COLORS['churn'],
           linestyle='--', alpha=0.8, label=f"Churn median: {df[df['Churn']==1]['tenure'].median():.0f}mo")

# 4. Monthly charges boxplot
ax = axes[1, 0]
churn_labels = {0: 'Retained', 1: 'Churned'}
data_box = [df[df['Churn'] == i]['MonthlyCharges'].values for i in [0, 1]]
bp = ax.boxplot(data_box, labels=['Retained', 'Churned'],
                patch_artist=True, notch=True)
bp['boxes'][0].set_facecolor(COLORS['retain'] + '80')
bp['boxes'][1].set_facecolor(COLORS['churn'] + '80')
ax.set_title('Monthly Charges by Churn', fontweight='bold')
ax.set_ylabel('Monthly Charges ($)')

# 5. Churn by Internet Service
ax = axes[1, 1]
ct2 = df.groupby('InternetService')['Churn'].mean().sort_values(ascending=False)
palette = [COLORS['churn'], '#F4A261', COLORS['retain']]
bars2 = ax.bar(ct2.index, ct2.values * 100, color=palette)
ax.set_title('Churn Rate by Internet Service', fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, 55)
for bar, val in zip(bars2, ct2.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

# 6. Churn by Payment Method
ax = axes[1, 2]
ct3 = df.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=False)
short_labels = [x.replace(' (automatic)', '').replace(' check', '\ncheck')
                for x in ct3.index]
bars3 = ax.bar(range(len(ct3)), ct3.values * 100,
               color=[COLORS['churn'], '#F4A261', COLORS['retain'], COLORS['retain']])
ax.set_xticks(range(len(ct3)))
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_title('Churn Rate by Payment Method', fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, 55)
for bar, val in zip(bars3, ct3.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: eda_plots.png")


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────

df_model = df.drop('customerID', axis=1).copy()

# encode binary categoricals
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df_model[col] = (df_model[col] == 'Yes').astype(int)

# encode multi-class categoricals
ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
            'TechSupport', 'Contract', 'PaymentMethod']
df_model = pd.get_dummies(df_model, columns=ohe_cols, drop_first=True)

# tenure bins as additional feature
df_model['tenure_group'] = pd.cut(df['tenure'],
                                   bins=[0, 12, 24, 48, 72],
                                   labels=[1, 2, 3, 4]).astype(int)

X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")
print(f"Features: {X.shape[1]}")


# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_sc, y_train)
lr_probs = lr.predict_proba(X_test_sc)[:, 1]
lr_preds = lr.predict(X_test_sc)
lr_auc = roc_auc_score(y_test, lr_probs)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                             random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_probs)

print(f"\nLogistic Regression  AUC: {lr_auc:.4f}")
print(f"Random Forest        AUC: {rf_auc:.4f}")
print(f"\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds, target_names=['Retained', 'Churned']))


# ─────────────────────────────────────────────
# 5. EVALUATION PLOTS
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Evaluation', fontsize=14, fontweight='bold')

# ROC Curve
ax = axes[0]
for probs, label, color in [
    (lr_probs, f'Logistic Regression (AUC={lr_auc:.3f})', '#457B9D'),
    (rf_probs, f'Random Forest (AUC={rf_auc:.3f})', '#E63946')
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, label=label, linewidth=2, color=color)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve', fontweight='bold')
ax.legend(fontsize=9)
ax.fill_between(*roc_curve(y_test, rf_probs)[:2], alpha=0.08, color='#E63946')

# Confusion Matrix (Random Forest)
ax = axes[1]
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r',
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'], ax=ax,
            cbar=False, linewidths=1)
ax.set_title('Confusion Matrix (Random Forest)', fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# Feature Importance (top 12)
ax = axes[2]
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
top12 = feat_imp.nlargest(12).sort_values()
colors_fi = ['#E63946' if v > top12.quantile(0.75) else '#457B9D' for v in top12]
top12.plot(kind='barh', ax=ax, color=colors_fi)
ax.set_title('Top 12 Feature Importances\n(Random Forest)', fontweight='bold')
ax.set_xlabel('Importance Score')
ax.axvline(top12.mean(), color='gray', linestyle='--', alpha=0.6, label='Mean')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: model_evaluation.png")


# ─────────────────────────────────────────────
# 6. BUSINESS INTERPRETATION
#    Risk segmentation for retention campaign
# ─────────────────────────────────────────────

results = X_test.copy()
results['Churn_Actual'] = y_test.values
results['Churn_Prob'] = rf_probs
results['Monthly_Charges'] = df.loc[X_test.index, 'MonthlyCharges'].values
results['Tenure'] = df.loc[X_test.index, 'tenure'].values

results['Risk_Segment'] = pd.cut(
    results['Churn_Prob'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

results['Estimated_Annual_Revenue'] = results['Monthly_Charges'] * 12
segment_summary = results.groupby('Risk_Segment', observed=True).agg(
    Customers=('Churn_Prob', 'count'),
    Avg_Churn_Prob=('Churn_Prob', 'mean'),
    Avg_Monthly_Charges=('Monthly_Charges', 'mean'),
    Revenue_at_Risk=('Estimated_Annual_Revenue', 'sum'),
    Actual_Churn_Rate=('Churn_Actual', 'mean')
).round(2)

print("\n--- BUSINESS RISK SEGMENTATION ---")
print(segment_summary.to_string())

high_risk = results[results['Risk_Segment'] == 'High Risk']
total_revenue_at_risk = high_risk['Estimated_Annual_Revenue'].sum()
print(f"\nHigh-Risk Customers: {len(high_risk):,}")
print(f"Annual Revenue at Risk: ${total_revenue_at_risk:,.0f}")
print(f"Average Churn Probability: {high_risk['Churn_Prob'].mean():.1%}")
print(f"\nRecommendation: Prioritize retention offers for {len(high_risk):,} high-risk")
print(f"customers with avg monthly charge of ${high_risk['Monthly_Charges'].mean():.2f}.")

print("\nAll outputs saved. See eda_plots.png and model_evaluation.png.")
