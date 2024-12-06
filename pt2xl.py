import torch
import pandas as pd

# TransformerDataset 클래스 정의 (로드를 위해 필수)
class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# .pt 파일 로드 함수
def pt_to_excel(pt_file_path, excel_file_path):
    # .pt 파일 로드
    data = torch.load(pt_file_path)

    # PyTorch Dataset 객체에서 데이터를 DataFrame으로 변환
    data_list = []
    for i in range(len(data)):
        # 샘플 구조를 확인 후 적절히 수정
        sample = data[i]  # 데이터셋의 각 샘플
        if isinstance(sample, tuple) or isinstance(sample, list):  # 튜플/리스트인 경우
            features, label = sample
            features = features.squeeze().tolist()  # 텐서를 리스트로 변환
            label = label.item()  # 텐서 값을 스칼라로 변환
        else:  # 샘플이 단일 값인 경우
            features = sample.squeeze().tolist()
            label = None  # 라벨이 없는 경우 처리
        data_list.append(features + ([label] if label is not None else []))  # 특징 + 라벨

    # 열 이름 정의 (입력 feature + label)
    columns = [f"feature_{i+1}" for i in range(len(features))]
    if label is not None:
        columns.append("label")
    df = pd.DataFrame(data_list, columns=columns)

    # Excel 파일로 저장
    df.to_excel(excel_file_path, index=False)
    print(f"Excel 파일로 저장되었습니다: {excel_file_path}")

# train_dataset.pt를 Excel로 변환
pt_to_excel("train_dataset.pt", "train_dataset.xlsx")

# test_dataset.pt를 Excel로 변환
pt_to_excel("test_dataset.pt", "test_dataset.xlsx")
