from matplotlib.pylab import randint
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

# 1. 데이터 불러오기
df = pd.read_csv('spo2_features_with_labels_fixed.csv', low_memory=False)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # or 'Arial'

# 2. 특징(X)과 라벨(y) 분리
X = df.drop(columns=["filename", "nsrrid", "ahi_a0h3", "apnea"])
y = df["apnea"]

# 3. 학습/테스트셋 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # +1) 샘플링 - smote
# smote = SMOTE(random_state=42)
# X_smo, y_smo = smote.fit_resample(X_train_scaled, y_train)

# # +1) 샘플링 - smoteTomek
# from imblearn.combine import SMOTETomek
# smt = SMOTETomek(random_state=42)
# X_resampled, y_resampled = smt.fit_resample(X_train_scaled, y_train)

# # +1) 샘플링 - Tomek
# from imblearn.under_sampling import TomekLinks
# t1 = TomekLinks()
# X_res, y_res = t1.fit_resample(X_train_scaled, y_train)

import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1200)
    max_depth = trial.suggest_int('max_depth', 4, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', [ 'sqrt', 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    # Stratified K-Fold를 활용한 AUC 평가
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    return auc_scores.mean()

# Optuna 튜닝 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("✅ Best Hyperparameters:")
print(study.best_params)

# 최적 모델 학습
best_rf = RandomForestClassifier(
    **study.best_params,
    random_state=42,
    n_jobs=-1
)
best_rf.fit(X_train_scaled, y_train)

# 5. 모델 정의
"""models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}"""

models = {
    # "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, class_weight='balanced'),
    "Random Forest": best_rf,   # 튜닝된 모델
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=1.9),
    "Support Vector Machine": SVC(probability=True, random_state=42, C=1.5, class_weight='balanced'),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7)
}  # 수정

# 6. 모델 학습 및 평가
'''
'''
results = [] # 원본 - 학습 데이터를 정규화로

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # 확률값으로 AUC 계산
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Precision (1)": report["1"]["precision"],
        "Recall (1)": report["1"]["recall"],
        "F1-score (1)": report["1"]["f1-score"],
        "AUC": auc
    })
"""
results = []  # 학습 데이터를 smote+Tomek로 적용

for name, model in models.items():
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # 확률값으로 AUC 계산
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Precision (1)": report["1"]["precision"],
        "Recall (1)": report["1"]["recall"],
        "F1-score (1)": report["1"]["f1-score"],
        "AUC": auc
    })
"""
'''
results = []  # 학습 데이터를 Tomek로 적용

for name, model in models.items():
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # 확률값으로 AUC 계산
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Precision (1)": report["1"]["precision"],
        "Recall (1)": report["1"]["recall"],
        "F1-score (1)": report["1"]["f1-score"],
        "AUC": auc
    })
'''

"""
results = []  # 학습 데이터를 smote로 적용

for name, model in models.items():
    model.fit(X_smo, y_smo)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # 확률값으로 AUC 계산
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Precision (1)": report["1"]["precision"],
        "Recall (1)": report["1"]["recall"],
        "F1-score (1)": report["1"]["f1-score"],
        "AUC": auc
    })
"""

# 7. 결과 출력
results_df = pd.DataFrame(results)
print("\n📊 모델 성능 비교:\n")
print(results_df)

# # ROC Curve 시각화
# plt.figure(figsize=(10, 8))
# for name, model in models.items():
#     y_proba = model.predict_proba(X_test_scaled)[:, 1]
#     fpr, tpr, _ = roc_curve(y_test, y_proba)
#     auc = roc_auc_score(y_test, y_proba)
#     plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

# plt.plot([0, 1], [0, 1], 'k--')
# plt.title("ROC Curve 비교")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # XGBoost Feature Importance
# xgb = models["XGBoost"]
# plt.figure(figsize=(10, 5))
# plt.barh(X.columns, xgb.feature_importances_)
# plt.title("XGBoost 특징 중요도")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.show()

# # Random Forest Feature Importance
# rf = models["Random Forest"]
# plt.figure(figsize=(10, 5))
# plt.barh(X.columns, rf.feature_importances_)
# plt.title("Random Forest 특징 중요도")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.show()