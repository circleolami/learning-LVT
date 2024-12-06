import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 데이터 준비
data_path = "data.xlsx"  # 데이터 경로
data = pd.read_excel(data_path, sheet_name="00-01-방역우선지DataSet확정본_211208")

# 필요한 열 선택
features = ['위도', '경도', '인구_총인구', '평균_토지_공시지가', '평균_토지대장_공시지가']
target = ['민원_발생여부_All']  # 타겟 변수 (해충 발생 여부)

# 결측값 처리
data = data.fillna(0)

# 입력과 타겟 데이터 분리
X = data[features]
y = data[target]

# 정규화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# PyTorch 데이터셋 정의
class PestDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]  # (input_dim) -> (1, input_dim)

train_dataset = PestDataset(X_train, y_train)
test_dataset = PestDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 트랜스포머 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout),
            num_layers=n_layers
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)
        x = self.transformer(x)  # (batch_size, seq_len, hidden_dim)
        x = x.mean(dim=1)  # (batch_size, hidden_dim)
        x = self.fc(x)  # (batch_size, hidden_dim) -> (batch_size, 1)
        return self.sigmoid(x)

# 모델 초기화
input_dim = len(features)
hidden_dim = 64
n_heads = 4
n_layers = 2

model = TransformerModel(input_dim, hidden_dim, n_heads, n_layers)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)  # X_batch: (batch_size, seq_len, input_dim)
            loss = criterion(y_pred.squeeze(), y_batch.squeeze())  # y_batch 크기 맞춤
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 평가 루프
def evaluate_model(model, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch).squeeze()
            y_pred_labels = (y_pred > 0.5).float()
            total_correct += (y_pred_labels == y_batch.squeeze()).sum().item()  # y_batch 크기 맞춤
            total_samples += y_batch.size(0)
    accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy:.4f}")

# 모델 학습
train_model(model, train_loader, criterion, optimizer, epochs=10)

# 모델 평가
evaluate_model(model, test_loader)
