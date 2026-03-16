import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score,
    classification_report, roc_curve, auc
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import seaborn as sns

df = pd.read_csv(
    'https://raw.githubusercontent.com/Huni-code/Loan-Approval-dataset/refs/heads/main/loan-train.csv'
)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

y = df['Loan_Status']
X = df.drop(columns=['Loan_ID', 'Loan_Status'])

print("=" * 60)
print("1. Data Setup")
print("=" * 60)

print("\nTable 1: Target Distribution")
print(y.value_counts().to_markdown(numalign='left', stralign='left'))

print("\nTable 2: Missing Values")
missing = df.isnull().sum()
missing = missing[missing > 0]
missing_df = pd.DataFrame({
    'feature': missing.index,
    'missing_count': missing.values,
    'missing_pct': (missing.values / len(df) * 100).round(2)
})
print(missing_df.to_markdown(index=False, numalign='left', stralign='left'))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
print("Table 3: Train Target Ratio")
print(
    y_train.value_counts(normalize=True)
    .round(4)
    .to_markdown(numalign='left', stralign='left')
)

numerical_features = [
    'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term',
    'Credit_History'
]

categorical_features = [
    'Gender', 'Married', 'Dependents',
    'Education', 'Self_Employed',
    'Property_Area'
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]),
            numerical_features
        ),
        (
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]),
            categorical_features
        )
    ]
)

print("\n" + "=" * 60)
print("2. Exploratory Data Analysis (EDA)")
print("=" * 60)

status_counts = df['Loan_Status'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(status_counts.index, status_counts.values,
        tick_label=['Rejected', 'Approved'])
plt.title('Fig 1: Loan Status Distribution (Imbalance Check)')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.savefig('fig1_loan_status_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

viz = df.copy()
viz['Credit_History_Viz'] = viz['Credit_History'].fillna('Missing').astype(str)

pd.crosstab(
    viz['Credit_History_Viz'],
    viz['Loan_Status']
).plot(kind='bar', figsize=(8, 6))
plt.title('Fig 2: Loan Status by Credit History (Class 0 Driver)')
plt.xlabel('Credit History (1.0 = Yes, 0.0 = No, Missing = NaN)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(['Rejected', 'Approved'], title='Loan Status')
plt.savefig('fig2_credit_history.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['ApplicantIncome'].plot(kind='hist', bins=40, ax=axes[0], edgecolor='black')
axes[0].set_title('Fig 3a: Applicant Income Distribution')
axes[0].set_xlabel('Income')
df['LoanAmount'].dropna().plot(kind='hist', bins=40, ax=axes[1], edgecolor='black')
axes[1].set_title('Fig 3b: Loan Amount Distribution')
axes[1].set_xlabel('Loan Amount')
plt.tight_layout()
plt.savefig('fig3_income_loanamt_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

numeric_df = df[numerical_features].copy()
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Fig 4: Correlation Heatmap (Numerical Features)')
plt.savefig('fig4_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("3. Model Comparison with Cross-Validation")
print("=" * 60)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']

cv_results = []

for name, model in models.items():
    pipe = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])

    scores = cross_validate(
        pipe, X_train, y_train,
        cv=cv, scoring=scoring, return_train_score=False
    )

    cv_results.append({
        'Model': name,
        'Accuracy': f"{scores['test_accuracy'].mean():.4f} +/- {scores['test_accuracy'].std():.4f}",
        'Precision': f"{scores['test_precision_macro'].mean():.4f} +/- {scores['test_precision_macro'].std():.4f}",
        'Recall': f"{scores['test_recall_macro'].mean():.4f} +/- {scores['test_recall_macro'].std():.4f}",
        'F1': f"{scores['test_f1_macro'].mean():.4f} +/- {scores['test_f1_macro'].std():.4f}",
        'ROC_AUC': f"{scores['test_roc_auc'].mean():.4f} +/- {scores['test_roc_auc'].std():.4f}"
    })

cv_df = pd.DataFrame(cv_results)
print("\nTable 4: 5-Fold Cross-Validation Results (with SMOTE)")
print(cv_df.to_markdown(index=False, numalign='left', stralign='left'))

print("\n" + "=" * 60)
print("4. Final Model Evaluation on Test Set")
print("=" * 60)

final_pipelines = {}
for name, model in models.items():
    pipe = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    pipe.fit(X_train, y_train)
    final_pipelines[name] = pipe

test_results = []
for name, pipe in final_pipelines.items():
    y_pred = pipe.predict(X_test)
    test_results.append({
        'Model': name,
        'Accuracy': f"{accuracy_score(y_test, y_pred):.4f}",
        'Precision_0': f"{precision_score(y_test, y_pred, pos_label=0, zero_division=0):.4f}",
        'Recall_0': f"{recall_score(y_test, y_pred, pos_label=0, zero_division=0):.4f}",
        'Precision_1': f"{precision_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}",
        'Recall_1': f"{recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}"
    })

test_df = pd.DataFrame(test_results)
print("\nTable 5: Test Set Results (with SMOTE)")
print(test_df.to_markdown(index=False, numalign='left', stralign='left'))

print("\n" + "=" * 60)
print("5. ROC-AUC Curves")
print("=" * 60)

plt.figure(figsize=(10, 7))
for name, pipe in final_pipelines.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    print(f"{name} ROC-AUC: {roc_auc:.4f}")

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.title('Fig 5: ROC Curves - Model Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('fig5_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("6. AdaBoost Detailed Analysis")
print("=" * 60)

adaboost_pipe = final_pipelines['AdaBoost']
y_pred_ada = adaboost_pipe.predict(X_test)

print("\nConfusion Matrix (AdaBoost + SMOTE):")
cm = confusion_matrix(y_test, y_pred_ada)
print(cm)

print(f"\nClassification Report (AdaBoost + SMOTE):")
print(classification_report(y_test, y_pred_ada, target_names=['Rejected', 'Approved']))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'])
plt.title('Fig 6: Confusion Matrix (AdaBoost + SMOTE)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('fig6_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

model_ada = adaboost_pipe.named_steps['classifier']
pre = adaboost_pipe.named_steps['preprocessor']

feature_names = pre.get_feature_names_out()
clean_names = [name.split('__')[-1] for name in feature_names]

feature_importances = pd.DataFrame({
    'feature': clean_names,
    'importance': model_ada.feature_importances_
})

top_10 = feature_importances.sort_values(
    by='importance', ascending=False
).head(10)

print("\nTable 6: Top 10 Important Features (AdaBoost + SMOTE)")
print(top_10.to_markdown(numalign='left', stralign='left'))

plt.figure(figsize=(10, 6))
plt.barh(
    top_10['feature'][::-1],
    top_10['importance'][::-1]
)
plt.title('Fig 7: Top 10 Feature Importances (AdaBoost + SMOTE)')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.savefig('fig7_feature_importances.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("7. Financial Risk Analysis")
print("=" * 60)

proba_scores = adaboost_pipe.predict_proba(X_test)
df_proba = pd.DataFrame(proba_scores, columns=['Rejected', 'Approved'])
df_proba['True_Label'] = y_test.values

threshold = 0.5
df_proba['Predicted_Label'] = (df_proba['Approved'] >= threshold).astype(int)

TP = df_proba[(df_proba['True_Label'] == 1) & (df_proba['Predicted_Label'] == 1)].shape[0]
FP = df_proba[(df_proba['True_Label'] == 0) & (df_proba['Predicted_Label'] == 1)].shape[0]
TN = df_proba[(df_proba['True_Label'] == 0) & (df_proba['Predicted_Label'] == 0)].shape[0]
FN = df_proba[(df_proba['True_Label'] == 1) & (df_proba['Predicted_Label'] == 0)].shape[0]

print(f"\nThreshold: {threshold}")
print(f"Safe Approvals (TP): {TP}")
print(f"Risky Approvals (FP): {FP}")
print(f"Correctly Rejected (TN): {TN}")
print(f"Missed Opportunities (FN): {FN}")

print("\n" + "=" * 60)
print("8. Threshold Tuning")
print("=" * 60)

y_proba_approved = adaboost_pipe.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0.25, 0.99, 100)

precisions_0, recalls_0, f1_scores_0 = [], [], []
precisions_1, recalls_1, f1_scores_1 = [], [], []

for t in thresholds:
    y_pred_loop = (y_proba_approved >= t).astype(int)
    report = classification_report(
        y_test, y_pred_loop,
        output_dict=True, zero_division=0
    )
    precisions_0.append(report['0']['precision'])
    recalls_0.append(report['0']['recall'])
    f1_scores_0.append(report['0']['f1-score'])
    precisions_1.append(report['1']['precision'])
    recalls_1.append(report['1']['recall'])
    f1_scores_1.append(report['1']['f1-score'])

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions_0, label='Precision (Class 0: Rejected)')
plt.plot(thresholds, recalls_0, label='Recall (Class 0: Rejected)')
plt.plot(thresholds, f1_scores_0, label='F1 Score (Class 0: Rejected)')
plt.title('Fig 8: Metrics vs Threshold (Class 0, Rejected)')
plt.xlabel('Threshold for Approval')
plt.ylabel('Score')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('fig8_threshold_class0.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions_1, label='Precision (Class 1: Approved)')
plt.plot(thresholds, recalls_1, label='Recall (Class 1: Approved)')
plt.plot(thresholds, f1_scores_1, label='F1 Score (Class 1: Approved)')
plt.title('Fig 9: Metrics vs Threshold (Class 1, Approved)')
plt.xlabel('Threshold for Approval')
plt.ylabel('Score')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('fig9_threshold_class1.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll figures saved (fig1 ~ fig9).")