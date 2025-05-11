# 🔧 필수 라이브러리 설치 필요 (한 번만 실행)
# pip install torch pandas scikit-learn matplotlib

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_csv(r'C:\Users\\henry\Desktop\spo2_features_with_labels_fixed.csv', low_memory=False)

# 2. X, y 분리
X = df.drop(columns=["filename", "nsrrid", "ahi_a0h3", "apnea"]).values
y = df["apnea"].values

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. PyTorch Dataset
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=64, shuffle=False)

# 6. MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 7. 학습 함수
def train(model, loader, optimizer, criterion):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch).squeeze()
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

# 8. 평가 함수
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = model(X_batch).squeeze()
            y_true.extend(y_batch.numpy())
            y_pred.extend(probs.numpy())
    return y_true, y_pred

# 9. 학습 실행
input_dim = X.shape[1]
model = MLP(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    train(model, train_loader, optimizer, criterion)
    y_true, y_pred = evaluate(model, test_loader)
    auc = roc_auc_score(y_true, y_pred)
    print(f"Epoch {epoch+1}/{epochs} - AUC: {auc:.4f}")

# 10. 최종 결과 출력
y_true_bin = [int(y) for y in y_true]
y_pred_bin = [1 if p >= 0.5 else 0 for p in y_pred]
print("\n📊 Classification Report:\n")
print(classification_report(y_true_bin, y_pred_bin))
print(f"AUC Score: {roc_auc_score(y_true_bin, y_pred):.4f}")
