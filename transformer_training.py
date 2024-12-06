import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 데이터 파일 불러오기
data_path = "data.xlsx"  # 데이터 파일 경로를 지정
data = pd.read_excel(data_path, sheet_name="00-01-방역우선지DataSet확정본_211208")

# 1단계: 모델 학습에 필요한 열만 선택
features = ['위도', '경도', '인구_총인구', '평균_토지_공시지가', '평균_토지대장_공시지가', '민원_발생여부_All']
data = data[features]

# 2단계: 결측값 처리 (결측값을 0으로 채움)
data = data.fillna(0)

# 3단계: 데이터 정규화 (값을 0~1 범위로 변환)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 정규화된 데이터를 DataFrame 형식으로 변환 (이름 유지)
scaled_data_df = pd.DataFrame(scaled_data, columns=features)

# 4단계: 데이터를 학습용(train)과 테스트용(test)으로 나누기
train_data, test_data = train_test_split(scaled_data_df, test_size=0.2, random_state=42)

# 5단계: 데이터를 PyTorch 텐서로 변환
train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
test_tensor = torch.tensor(test_data.values, dtype=torch.float32)

# PyTorch 데이터셋 클래스 정의
class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 학습용 및 테스트용 데이터셋 생성
train_dataset = TransformerDataset(train_tensor)
test_dataset = TransformerDataset(test_tensor)

# 6단계: DataLoader 준비 (미니 배치로 데이터를 제공)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 데이터셋 저장
torch.save(train_dataset, "train_dataset.pt")  # 학습용 데이터셋 저장
torch.save(test_dataset, "test_dataset.pt")  # 테스트용 데이터셋 저장

print("데이터셋이 저장되었습니다: train_dataset.pt, test_dataset.pt")
