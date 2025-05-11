import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# 1. 데이터 불러오기
df = pd.read_csv(r'C:\Users\\henry\Desktop\spo2_features_with_labels_fixed.csv', low_memory=False)
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

# 5. 모델 정의
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# 6. 모델 학습 및 평가
results = []

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

# 7. 결과 출력
results_df = pd.DataFrame(results)
print("\n📊 모델 성능 비교:\n")
print(results_df)

# ROC Curve 시각화
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve 비교")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# XGBoost Feature Importance
xgb = models["XGBoost"]
plt.figure(figsize=(10, 5))
plt.barh(X.columns, xgb.feature_importances_)
plt.title("XGBoost 특징 중요도")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Random Forest Feature Importance
rf = models["Random Forest"]
plt.figure(figsize=(10, 5))
plt.barh(X.columns, rf.feature_importances_)
plt.title("Random Forest 특징 중요도")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()