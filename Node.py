# NODE-like 구조 구현 (MLP + Gating Ensemble)

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# 1. 데이터 불러오기
df = pd.read_csv(r"C:\Users\henry\Desktop\spo2_features_with_labels_fixed.csv")
X = df.drop(columns=["filename", "nsrrid", "ahi_a0h3", "apnea"]).values
y = df["apnea"].values

# 2. 데이터 분할 및 정규화
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Dataset 정의
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

# 4. 간이 NODE 구조
class NODE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_trees=4):
        super().__init__()
        self.trees = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_trees)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_trees),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        tree_outputs = torch.stack([tree(x).squeeze(1) for tree in self.trees], dim=1)
        weights = self.gate(x)
        return torch.sum(tree_outputs * weights, dim=1)

# 5. 학습/평가
model = NODE(input_dim=X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = model(X_batch)
            y_true.extend(y_batch.numpy())
            y_pred.extend(probs.numpy())
    return y_true, y_pred

# 6. 학습 루프
for epoch in range(20):
    train(model, train_loader)
    y_true, y_pred = evaluate(model, test_loader)
    auc = roc_auc_score(y_true, y_pred)
    print(f"Epoch {epoch+1}/20 - AUC: {auc:.4f}")

# 7. 최종 결과
y_pred_bin = [1 if p >= 0.5 else 0 for p in y_pred]
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_bin))
print(f"AUC Score: {roc_auc_score(y_true, y_pred):.4f}")
