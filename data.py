import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 문서 갖고오기
df = pd.read_csv('shhs1-dataset-0.20.0.csv')

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
df_selected = remove_outliers_iqr(df_selected, ['avgsat','minsat','pctsa85h','pctsa90h','ahi_a0h3'])
df_selected.fillna(df_selected.mode().iloc[0], inplace=True)

# 데이터 분리 (x,y,train/test)
x = df_selected.drop(columns=['ahi_a0h3','apnea'])  # 특징 (features, 모델이 예측할 입력 데이터) - 개인 식별자, 불면증 여부, label을 제거한 값들이 x로 입력
y = df_selected['apnea'] # 라벨 (예측해야 하는 결과값)

# 학습/테스트 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,stratify=y, random_state=12)

# 스케일링/정규화  // standardScalar 사용
scalar = StandardScaler()
scaled_data = scalar.fit_transform(df_selected)

print("Standardization 적용 후 데이터: \n", scaled_data)

# 인코딩

# 샘플링

# 모델 학습 