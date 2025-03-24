# %%
import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

# %%
# Определение устройства (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
# Определение класса датасета
class AudioDataset(Dataset):
    def __init__(self, root_dir, sr=16000, n_mfcc=40, max_len=220):
        self.file_paths = []  # список путей к аудиофайлам
        self.labels = []      # соответствующие метки классов
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len  # максимальная длина для паддинга
        self._load_dataset(root_dir)

    def _load_dataset(self, root_dir):
        class_labels = os.listdir(root_dir)
        for label_idx, class_label in enumerate(class_labels):
            class_dir = os.path.join(root_dir, class_label)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(class_dir, file_name)
                        self.file_paths.append(file_path)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(file_path, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

        # Паддинг или обрезка
        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = F.pad(torch.tensor(mfcc, dtype=torch.float32), (0, pad_width))
        else:
            mfcc = torch.tensor(mfcc, dtype=torch.float32)[:, :self.max_len]

        mfcc = mfcc.unsqueeze(0)
        return mfcc, label

# %%
# Определение модели
class VoiceRecognitionCNN(nn.Module):
    def __init__(self, num_classes):
        super(VoiceRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = None  # определим позже
        self.fc2 = None  # определим позже
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.fc1 is None or self.fc2 is None:
            num_features = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(num_features, 128).to(device)
            self.fc2 = nn.Linear(128, self.num_classes).to(device)
        x = x.view(x.size(0), -1)  # динамически определяем
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# %%
# Использование
dataset = AudioDataset(root_dir='data_audio')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

num_classes = len(set(dataset.labels))
model = VoiceRecognitionCNN(num_classes).to(device)  # Перемещаем модель на устройство

# Определяем функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# Тренировка модели
for epoch in range(5):  # 5 эпох
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)  # Перемещаем данные на устройство
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# %%
# Создание тестового датасета
test_dataset = AudioDataset(root_dir='test_data_audio')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
correct = 0
total = 0

# %%
# Без градиентов, чтобы увеличить производительность и сэкономить память
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)  # Перемещаем данные на устройство
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test data: {accuracy}%')

# %%
# Собираем все истинные и предсказанные метки
true_labels = []
pred_labels = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)  # Перемещаем данные на устройство
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        true_labels.extend(labels.cpu().numpy())  # Перемещаем обратно на CPU для обработки
        pred_labels.extend(predicted.cpu().numpy())

print("Classification Report:")
print(classification_report(true_labels, pred_labels))

print("Confusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))