from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# 데이터 불러오기
df = pd.read_csv(r'C:\Users\henry\Desktop\spo2_features_with_labels_fixed.csv')

# X, y 분리
X = df.drop(columns=["filename", "nsrrid", "ahi_a0h3", "apnea"]).values
y = df["apnea"].values

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 표준화 (TabNet은 로그 변환이나 정규화가 없어도 잘 작동하지만 비교를 위해 사용)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TabNet 모델 정의 및 학습
tabnet = TabNetClassifier(verbose=1, seed=42)
tabnet.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_test, y_test)],
    eval_name=["valid"],
    eval_metric=["auc"],
    max_epochs=100,
    patience=10,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
)

# 예측
y_proba = tabnet.predict_proba(X_test)[:, 1]
y_pred = tabnet.predict(X_test)

# 평가
auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)
print("\n📊 TabNet 성능:\n")
print(report)
print(f"AUC: {auc:.4f}")
