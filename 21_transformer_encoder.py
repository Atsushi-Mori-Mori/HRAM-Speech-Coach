# 　-*- coding: utf-8 -*-
import sys
import os
import re
import struct
import binascii
import numpy as np
from math import floor, ceil
import shutil
# -------------------------------------------------------
# 1. ライブラリーのインポート, データ読込み
# ライブラリーのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import glob
# import lightgbm as lgb
# -------------------------------------------------------
import matplotlib.pyplot as plt
# import cv2
# from torchvision import transforms
import numpy as np
# from pydub import AudioSegment
# # -------------------------------------------------------
### データセット作成
# # -------------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import glob

# カスタムデータセットクラス（学習用、ラベル付き）
class CsvDatasetWithLabels(Dataset):
    def __init__(self, data_folder, label_file):
        self.data_folder = data_folder
        self.labels_df = pd.read_csv(label_file)
        self.filepaths = glob.glob(os.path.join(data_folder, "*.csv"))
        self.labels = []
        for filepath in self.filepaths:
            filename = os.path.basename(filepath)
            label = self.get_label(filename)
            self.labels.append(label)

    def get_label(self, filename):
        label_row = self.labels_df[self.labels_df['filename'] == filename]
        if not label_row.empty:
            return label_row['label'].values[0]
        else:
            raise ValueError(f"ラベルが見つかりません: {filename}")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        data = pd.read_csv(filepath).values.astype(float)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return data_tensor, label_tensor

# カスタムデータセットクラス（テスト用、ラベルなし）
class CsvDatasetWithoutLabels(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.filepaths = glob.glob(os.path.join(data_folder, "*.csv"))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        data = pd.read_csv(filepath).values.astype(float)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        return data_tensor

# Transformerベースの分類モデルの定義
class TransformerClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes):
        super(TransformerClassificationModel, self).__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        output = self.fc(x)
        return output

# モデルのトレーニング関数
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_data, batch_labels in train_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        batch_data = batch_data.permute(1, 0, 2)  # データの形を (sequence_length, batch_size, input_dim) に変換
        optimizer.zero_grad()
        output = model(batch_data)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# モデルの評価関数
def evaluate_model(val_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            batch_data = batch_data.permute(1, 0, 2)
            output = model(batch_data)
            _, predicted = torch.max(output, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    return correct / total * 100  # 正解率

# クロスバリデーションの実行
def cross_validate(train_dataset, k_folds, model_class, input_dim, hidden_dim, num_heads, num_layers, num_classes, learning_rate, num_epochs, device):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold+1}/{k_folds}')

        # データセットの分割
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

        # モデルの初期化
        model = model_class(input_dim, hidden_dim, num_heads, num_layers, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # トレーニング
        for epoch in range(num_epochs):
            train_loss = train_model(train_loader, model, criterion, optimizer, device)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')

        # 評価
        accuracy = evaluate_model(val_loader, model, device)
        print(f'Fold {fold+1} Accuracy: {accuracy:.2f}%')
        fold_accuracies.append(accuracy)

    # クロスバリデーションの平均精度
    mean_accuracy = np.mean(fold_accuracies)
    print(f'Mean Accuracy across {k_folds} folds: {mean_accuracy:.2f}%')
    return mean_accuracy

# 最終学習とテスト実行
def final_training_and_testing(train_dataset, test_loader, model_class, input_dim, hidden_dim, num_heads, num_layers, num_classes, learning_rate, num_epochs, device):
    # 全データを使って再学習
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = model_class(input_dim, hidden_dim, num_heads, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学習
    for epoch in range(num_epochs):
        train_loss = train_model(train_loader, model, criterion, optimizer, device)
        print(f'Final Training Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')

    # テスト推論
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            batch_data = batch_data.permute(1, 0, 2)
            output = model(batch_data)
            predicted_class = torch.argmax(output, dim=1).cpu().numpy()
            predictions.append(predicted_class[0])

    return predictions

# データセットの準備
train_data_folder = "./data_csv/train"
test_data_folder = "./data_csv/test"
label_file = "./label.csv"
train_dataset = CsvDatasetWithLabels(train_data_folder, label_file)
test_dataset = CsvDatasetWithoutLabels(test_data_folder)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 23次元学習用の設定--------------------------------
# input_dim = 23  # 23個の特徴量
# # input_dim = 13  # 13個の特徴量(pText削除)
# hidden_dim = 64  # hidden_dimはnum_headsで割り切れる値にする
# num_heads = 1    # num_headsをinput_dimの約数にする（例：2）
# # num_heads = 7    # num_headsをinput_dimの約数にする（例：2）
# # num_layers = 4
# num_layers = 10
# num_classes = 4  # 4値分類 (0, 1, 2, 3)
# # learning_rate = 0.001
# learning_rate = 0.001
# -----------------------------------------------
# 24次元学習用の設定--------------------------------
input_dim = 18  # 18個の特徴量
hidden_dim = 64  # hidden_dimはnum_headsで割り切れる値にする(Input X 4程度)
# hidden_dim = 96  # hidden_dimはnum_headsで割り切れる値にする(Input X 4程度)
num_heads = 3    # num_headsをinput_dimの約数にする（例：2）
# num_heads = 6    # num_headsをinput_dimの約数にする（例：2）
# num_layers = 4
num_layers = 10
num_classes = 4  # 4値分類 (0, 1, 2, 3)
# learning_rate = 0.001
learning_rate = 0.001
# -----------------------------------------------

num_epochs = 16
k_folds = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# クロスバリデーション
cross_validate(train_dataset, k_folds, TransformerClassificationModel, input_dim, hidden_dim, num_heads, num_layers, num_classes, learning_rate, num_epochs, device)

# 全データで再学習とテスト推論
predictions = final_training_and_testing(train_dataset, test_loader, TransformerClassificationModel, input_dim, hidden_dim, num_heads, num_layers, num_classes, learning_rate, num_epochs, device)

# # テスト結果と比較して正解率を計算
results_df = pd.read_csv('./results.csv')  # 正解のテスト結果が書かれたCSV
true_labels = results_df['label'].values  # 正解ラベル
pred_labels = predictions  # モデルの予測結果

# # 正解率の計算
accuracy = sum(true_labels == pred_labels) / len(true_labels) * 100
print(f'Test Accuracy: {accuracy:.2f}%')
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------
# # -------------------------------------------------------

