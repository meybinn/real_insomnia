import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# 문서 갖고오기
df = pd.read_csv(r'C:\Users\\henry\Desktop\shhs1-dataset-0.20.0.csv', low_memory=False)

# 주요 변수 선정 // 산소포화도 평균&최소, saturation<90/85 횟수(percentage), ahi
selected_columns = ['avgsat','minsat','pctsa85h','pctsa90h','ahi_a0h3']

df_selected = df[selected_columns].copy()

df_selected.to_csv('shhs_sleep_selected.csv',index=False)

# print("!!!!!!!!!!!!!!!!완료!!!!!!!!!!!!!!!!") # 변수 처리까지

# label // ahi를 label로 정의
df_selected['apnea'] = df_selected['ahi_a0h3'].apply(lambda x:1 if(x>=15)else 0)

# print("!!!!!!!!!!!!",df_selected["apnea"].value_counts(),"!!!!!!!!!!!!")

# print("!!!!!!!!!!!!",df_selected["apnea"].value_counts(),"!!!!!!!!!!!!")

# 결측치 처리
  # 결측치 처리 - 결측치 확인
# print(df_selected.isna().sum()/len(df_selected))  # 결측치 비율 확인 -> 상태 양호            
# print("null ",df_selected.isnull().sum())  # 확인 결과 df_selected[minsat]의 결측치는 모두 null임
  # 결측치 처리 - null값들 처리
df_selected.fillna(df_selected.mode().iloc[0],inplace=True) # 결측치를 각 열의 최빈값으로 채움
# df_selected.fillna(df_selected.median(),inplace=True)  # 결측치를 평균값으로 채움

# 이상치 처리  // IQR적용
def remove_outliers_iqr(df,columns):
    for col in columns:
        Q1 = df[col].quantile(0.25) #1사분위
        Q3 = df[col].quantile(0.75) #3사분위
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR #박스플롯ㅉㅉ
        upper_bound = Q3 + 1.5 * IQR 
        # 이상치 바깥값을 NaN으로 설정 후 최빈값으로 대체 (삭제 또는 다른 방식도 가능)
        df[col] = df[col].apply(lambda x:np.nan if x < lower_bound or x > upper_bound else x)
        print(f"{col} 이상치 기준: < {lower_bound:.2f} 또는 > {upper_bound:.2f}")
    return df
  # 이상치 처리 - 이상치 제거
df_selected = remove_outliers_iqr(df_selected, ['avgsat','minsat','pctsa85h','pctsa90h','ahi_a0h3']) # pctsa85h 수치조정
df_selected.fillna(df_selected.mode().iloc[0], inplace=True)

# 특징 확장 (파생 변수 추가)
df_selected['avg_min_ratio'] = df_selected['avgsat'] / (df_selected['minsat'] + 1e-5)
df_selected['pct_diff_90_85'] = df_selected['pctsa90h'] - df_selected['pctsa85h']
df_selected['avg_minus_min'] = df_selected['avgsat'] - df_selected['minsat']


# 데이터 분리 (x,y,train/test)
X = df_selected.drop(columns=['ahi_a0h3','apnea'])  # 특징 (features, 모델이 예측할 입력 데이터) - 개인 식별자, 불면증 여부, label을 제거한 값들이 x로 입력
y = df_selected['apnea'] # 라벨 (예측해야 하는 결과값)

# 학습/테스트 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y, random_state=12)

# 스케일링/정규화  // standardScalar 사용
# scaled_data = scalar.fit_transform(df_selected)
# print("Standardization 적용 후 데이터: \n", scaled_data
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

# SMOTE 적용 (훈련셋만)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# SVM 모델 정의 + GridSearchCV로 하이퍼파라미터 튜닝
param_grid_svm = {
    'C': [0.1, 1, 10],               # SVM의 C 파라미터 (규제 강도): 작을수록 마진이 넓음, 클수록 오차를 줄이려 함
    'gamma': ['scale', 0.01, 0.1, 1],# gamma는 RBF 커널의 영향 범위 조절: 클수록 개별 샘플의 영향이 큼
    'kernel': ['rbf'],               # 커널 함수로 RBF(가우시안)만 사용 (비선형 분류에 강함)
    'class_weight': ['balanced']
}

grid_svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=5, n_jobs=-1)
grid_svm.fit(X_train_res, y_train_res)
print("\n=== SVM 최적 파라미터 ===")
print(grid_svm.best_params_)

svm_pred = grid_svm.predict(X_test_scaled)
print("\n=== SVM 결과 ===")
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# 🔥 3️⃣ RandomForest 하이퍼파라미터 튜닝
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'class_weight': ['balanced']
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=12), param_grid_rf, cv=5, n_jobs=-1)
grid_rf.fit(X_train_res, y_train_res)
print("\n=== RandomForest 최적 파라미터 ===")
print(grid_rf.best_params_)

# 🔥 4️⃣ XGBoost 하이퍼파라미터 튜닝
scale_pos_weight = len(y_train_res[y_train_res == 0]) / len(y_train_res[y_train_res == 1])
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1],
    'scale_pos_weight': [1, scale_pos_weight]
}
grid_xgb = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=12), 
                        param_grid_xgb, cv=5, n_jobs=-1)
grid_xgb.fit(X_train_res, y_train_res)
print("\n=== XGBoost 최적 파라미터 ===")
print(grid_xgb.best_params_)

# 🔥 5️⃣ 앙상블 모델 (Voting)
ensemble = VotingClassifier(
    estimators=[
        ('svm', grid_svm.best_estimator_),
        ('rf', grid_rf.best_estimator_),
        ('xgb', grid_xgb.best_estimator_)
    ],
    voting='soft'
)
ensemble.fit(X_train_res, y_train_res)

# 예측 및 평가 함수
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} 결과 ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.3f}")

# 🔥 최종 평가
evaluate_model('SVM', grid_svm.best_estimator_, X_test_scaled, y_test)
evaluate_model('RandomForest', grid_rf.best_estimator_, X_test_scaled, y_test)
evaluate_model('XGBoost', grid_xgb.best_estimator_, X_test_scaled, y_test)
evaluate_model('Ensemble (Voting)', ensemble, X_test_scaled, y_test)

# 인코딩

# 샘플링

# 모델 학습 