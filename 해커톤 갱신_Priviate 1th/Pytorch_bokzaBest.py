import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


# CSV 파일 읽기
df = pd.read_csv('test.csv')

# test.csv 'target' 열 추가
df['target'] = ''

# 결과를 새로운 파일로 저장
df.to_csv('new_test.csv', index=False)


class CustomDataset(Dataset):
    def __init__(self, dataframe, target_column='target', transform=None):
        self.dataframe = dataframe
        self.target_column = target_column
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = torch.tensor(self.dataframe.iloc[idx].drop(columns=[self.target_column]).values, dtype=torch.float32)
        target = torch.tensor(self.dataframe.iloc[idx][self.target_column], dtype=torch.float32)

        if self.transform:
            features = self.transform(features)

        return {'features': features, 'target': target}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(142)

import warnings
warnings.filterwarnings('ignore')

# 데이터 불러오기
train_org = pd.read_csv('train.csv')
test_org = pd.read_csv('new_test.csv')


# 데이터 전처리
train_df = train_org.copy()
test_df = test_org.copy()

categorical_features = list(train_df.dtypes[train_df.dtypes == "object"].index)


# Target Encoding
for i in categorical_features:
    le = TargetEncoder(cols=[i])
    train_df[i] = le.fit_transform(train_df[i], train_df['target'])
    test_df[i] = le.fit_transform(test_df[i], test_df['target'])

# 결측치 처리
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# 데이터셋 및 데이터로더 생성
train_dataset = CustomDataset(train_df)
test_dataset = CustomDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=10000, num_workers = 4, shuffle=True, pin_memory = True)
test_loader = DataLoader(test_dataset, batch_size=10000, num_workers = 4, shuffle=False, pin_memory = True)

class CustomModel(nn.Module):
    def __init__(self, input_size):
        super(CustomModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 128)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(128, 256)
        self.relu5 = nn.ReLU()
        self.output_layer1 = nn.Linear(256, 64)
        self.output_layer2 = nn.Linear(64, 1)

    def forward(self, x):
       # x = self.batch_norm(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.output_layer1(x)
        x = self.output_layer2(x)
        x = torch.sigmoid(x)  # Sigmoid 활성화 함수 추가
        return x

input_size = len(train_df.columns)  # 입력 피처의 개수
model = CustomModel(input_size)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005)

if __name__ == '__main__':
    # 학습
    num_epochs = 100
    print_interval = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features, target = batch['features'], batch['target']

            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target.view(-1, 1).float())  # MSELoss에 대한 타입 호환성 수정
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}')

    # 평가 및 예측
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            features, _ = batch['features'], batch['target']  # test 데이터셋에는 target이 필요하지 않음

            output = model(features)
            binary_output = torch.round(output)
            all_predictions.append(binary_output.cpu().numpy())

    # test 데이터셋에 대한 예측 결과를 하나의 배열로 통합
    all_predictions = np.concatenate(all_predictions)

    # 결과를 DataFrame으로 만들어 CSV 파일로 저장
    submission = pd.read_csv('sample_submission.csv')
    submission['target'] = all_predictions.astype(int)  # 정수형으로 변환하여 저장
    # submission.to_csv(f'submission_torchMLP_1204.csv', index=False)
    submission.to_csv(f'Best_test5.csv', index=False)
    print(submission)

