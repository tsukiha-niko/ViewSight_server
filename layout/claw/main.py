import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# 1. 数据加载和预处理
data = pd.read_csv('info.csv', encoding='utf-8-sig')  # 确保 UTF-8 读取，防止乱码

# 检查数据类型
print("Data types before conversion:")
print(data.dtypes)

# 假设列名为：fan_count, prev_rating, prev_view, title, cover, video_rating
numeric_cols = ['fan_count', 'prev_rating', 'prev_view']

# **确保数值型列全部转换为 float**
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')  # 处理可能的字符串
data['video_rating'] = pd.to_numeric(data['video_rating'], errors='coerce')  # 目标列也要转换

# **检查是否有 NaN 数据**
if data[numeric_cols].isnull().values.any():
    print("警告：数值列存在 NaN 值，正在填充 0")
    data[numeric_cols] = data[numeric_cols].fillna(0)
if data['video_rating'].isnull().values.any():
    print("警告：目标值 video_rating 存在 NaN，正在填充均值")
    data['video_rating'] = data['video_rating'].fillna(data['video_rating'].mean())

# **标准化数值特征**
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 再次检查数据类型
print("Data types after conversion:")
print(data.dtypes)


# 2. 定义数据集
class VideoDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # **确保转换数值型数据**
        numeric = torch.tensor(row[numeric_cols].astype(float).values, dtype=torch.float)

        # 文本特征（标题与封面）
        title = str(row['title'])
        cover = str(row['cover'])
        combined_text = title + ' [SEP] ' + cover
        encoding = self.tokenizer(combined_text,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # **确保目标值转换为 float**
        target = torch.tensor(float(row['video_rating']), dtype=torch.float)

        return {
            'numeric': numeric,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': target
        }


# 使用中文预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
dataset = VideoDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# 3. 定义模型
class MultiModalPredictor(nn.Module):
    def __init__(self, hidden_size=768, numeric_size=3, fusion_size=256):
        super(MultiModalPredictor, self).__init__()
        # 文本编码器
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # 数值特征全连接
        self.numeric_fc = nn.Sequential(
            nn.Linear(numeric_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        # 融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_size + 32, fusion_size),
            nn.ReLU(),
            nn.Linear(fusion_size, 1)  # 输出为一个评分数值
        )

    def forward(self, input_ids, attention_mask, numeric):
        # 文本特征
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        # 数值特征
        numeric_feat = self.numeric_fc(numeric)
        # 融合
        fused = torch.cat([text_feat, numeric_feat], dim=1)
        output = self.fusion_fc(fused)
        return output.squeeze(1)  # (batch_size)


# 实例化模型并使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiModalPredictor().to(device)

# 4. 训练设置
criterion = nn.MSELoss()  # 评分是连续数值，用均方误差作为损失函数
optimizer = optim.Adam(model.parameters(), lr=2e-5)

num_epochs = 5  # 训练 5 轮

# 5. 训练循环
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numeric = batch['numeric'].to(device)
        targets = batch['target'].to(device)

        outputs = model(input_ids, attention_mask, numeric)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs} Loss: {total_loss / len(dataloader):.4f}")

# 6. 保存模型
torch.save(model.state_dict(), 'teacher_model.pt')
print("模型已保存为 teacher_model.pt")
