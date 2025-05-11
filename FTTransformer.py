# FTTransformer (PyTorch 기반 간이 구조)

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import Dataset, DataLoader

# 1. 데이터 로딩
df = pd.read_csv(r"C:\Users\henry\Desktop\spo2_features_with_labels_fixed.csv")
X = df.drop(columns=["filename", "nsrrid", "ahi_a0h3", "apnea"]).values
y = df["apnea"].values

# 2. 데이터 분할 및 정규화
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. PyTorch Dataset
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

# 4. FTTransformer 구조
class FTTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.output(x)

# 5. 학습 함수
def train(model, loader, optimizer, criterion):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch).squeeze()
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = model(X_batch).squeeze()
            y_true.extend(y_batch.numpy())
            y_pred.extend(probs.numpy())
    return y_true, y_pred

# 6. 학습
model = FTTransformer(input_dim=X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(20):
    train(model, train_loader, optimizer, criterion)
    y_true, y_pred = evaluate(model, test_loader)
    auc = roc_auc_score(y_true, y_pred)
    print(f"Epoch {epoch+1}/20 - AUC: {auc:.4f}")

# 7. 최종 결과
y_pred_bin = [1 if p >= 0.5 else 0 for p in y_pred]
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred_bin))
print(f"AUC Score: {roc_auc_score(y_true, y_pred):.4f}")
