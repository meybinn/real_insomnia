import pandas as pd
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

print("!!!!!!!!!!!!",df_selected["apnea"].value_counts(),"!!!!!!!!!!!!")

# 결측치 처리

  # 결측치 처리 - 결측치 확인
# print(df_selected.isna().sum()/len(df_selected))  # 결측치 비율 확인 -> 상태 양호
# print("null ",df_selected.isnull().sum())  # 확인 결과 df_selected[minsat]의 결측치는 모두 null임

  # 결측치 처리 - null값들 처리
df_selected.fillna(df_selected.mode().iloc[0],inplace=True) # 결측치를 각 열의 최빈값으로 채움
# df_selected.fillna(df_selected.median(),inplace=True)  # 결측치를 평균값으로 채움

# 이상치 처리


# 스케일링



# 모델 학습