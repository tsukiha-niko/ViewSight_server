import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from snownlp import SnowNLP
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# 设置随机种子和设备
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train_v21_fallback.py] Using device: {device}")

# 目录结构
TEMP_DIR = "./temp"
MODEL_DIR = "./models"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#############################################################
# 1) 数据加载与预处理
#############################################################
df_raw = pd.read_csv('merged.csv')  # 加载原始数据
print(f"Loaded {len(df_raw)} rows of data from 'merged.csv'.")

print("=== 播放数分布概览（包含可能的极端值） ===")
print(df_raw['播放数'].describe())

# === 进行轻度异常值裁剪，减少极端大播放数对训练的干扰 ===
df_raw['播放数'] = np.clip(df_raw['播放数'], 0, 1e8)
print("After capping, basic statistics of 播放数:\n", df_raw['播放数'].describe())

### 过滤掉播放量为 0 的项
df = df_raw[df_raw['播放数'] > 0].copy().reset_index(drop=True)
print(f"After removing 0-play videos, remaining data size = {len(df)}")

# 数值特征：粉丝数、p_rating、时长（秒）、trending、emotion、visual、creativity
numeric_cols = ['粉丝数', 'p_rating', '时长（秒）', 'trending', 'emotion', 'visual', 'creativity']
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(df[numeric_cols].fillna(0).values)  # 缺失值填 0
joblib.dump(scaler, os.path.join(TEMP_DIR, 'scaler_v21.pkl'))

# 目标：播放数（对数+1）
y = np.log1p(df['播放数'].values)

# 分类辅助：播放数>100 => 1，否则0
y_cls = (df['播放数'] > 100).astype(float).values

# 标题情感
sentiments = np.array([
    SnowNLP(t).sentiments if isinstance(t, str) else 0.5
    for t in df['视频标题']
]).reshape(-1, 1)

sentiment_scaler = StandardScaler()
sentiments_scaled = sentiment_scaler.fit_transform(sentiments)
joblib.dump(sentiment_scaler, os.path.join(TEMP_DIR, 'sentiment_scaler_v21.pkl'))

# 数值特征拼起来 => 7列（粉丝数等） + 1列情感 + 1列播放分类 => 9列
X_numeric = np.hstack([X_numeric_scaled, sentiments_scaled, y_cls.reshape(-1, 1)])

#############################################################
# 2) 加载/生成 BERT 向量（基于过滤后的数据）
#############################################################
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
bert_model.eval()


def text_to_vector(text):
    """将文本转换成 [CLS] 向量。"""
    if not isinstance(text, str):
        text = ""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=64
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


def get_vectors(df_col, filename, desc, valid_indices):
    filepath = os.path.join(TEMP_DIR, filename)
    if os.path.exists(filepath):
        print(f"Loading cached BERT vectors for {df_col} from {filepath} ...")
        vectors_full = np.load(filepath)
        vectors = vectors_full[valid_indices]  # 筛选与过滤后数据匹配的向量
        if vectors.shape[0] != len(valid_indices):
            print(
                f"Warning: Cached vectors size ({vectors.shape[0]}) does not match filtered data size ({len(valid_indices)}). Regenerating...")
            vectors_list = [text_to_vector(txt) for txt in tqdm(df[df_col], desc=desc)]
            vectors = np.stack(vectors_list)
            np.save(filepath, vectors)
    else:
        print(f"Generating BERT vectors for {df_col} ...")
        vectors_list = [text_to_vector(txt) for txt in tqdm(df[df_col], desc=desc)]
        vectors = np.stack(vectors_list)
        np.save(filepath, vectors)
    return vectors


# 获取过滤后的有效索引（与 df 对应）
valid_indices = df.index.values  # 已经是过滤后的索引，因为 df 已重置

title_vectors = get_vectors('视频标题', 'title_vectors_v21.npy', "Titles", valid_indices)
cover_vectors = get_vectors('封面评论', 'cover_vectors_v21.npy', "Covers", valid_indices)
partition_vectors = get_vectors('分区', 'partition_vectors_v21.npy', "Partitions", valid_indices)


#############################################################
# 3) PCA 降维
#############################################################
def get_pca(vectors, filename, n_components=256):
    filepath = os.path.join(TEMP_DIR, filename)
    if os.path.exists(filepath):
        pca = joblib.load(filepath)
    else:
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(vectors)
        joblib.dump(pca, filepath)
    return pca.transform(vectors)


title_pca = get_pca(title_vectors, 'pca_title_v21.pkl')
cover_pca = get_pca(cover_vectors, 'pca_cover_v21.pkl')
partition_pca = get_pca(partition_vectors, 'pca_partition_v21.pkl')

X_text = np.hstack([title_pca, cover_pca, partition_pca])  # 768维 (3*256)
X = np.hstack([X_text, X_numeric])  # 最终 777维 (768 + 9)

#############################################################
# 4) 切分训练集/验证集
#############################################################
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)


#############################################################
# 5) 定义模型
#############################################################
class PlayCountPredictorV21(nn.Module):
    def __init__(self, input_dim, text_dim=256 * 3):
        super(PlayCountPredictorV21, self).__init__()
        self.text_fc1 = nn.Linear(text_dim, 512)
        self.text_bn1 = nn.BatchNorm1d(512)
        self.text_fc2 = nn.Linear(512, 256)
        self.text_bn2 = nn.BatchNorm1d(256)
        self.text_attn = nn.Linear(256, 256)

        self.other_fc1 = nn.Linear(input_dim - text_dim, 256)  # 9 维数值特征
        self.other_bn1 = nn.BatchNorm1d(256)
        self.other_fc2 = nn.Linear(256, 128)
        self.other_bn2 = nn.BatchNorm1d(128)

        self.fusion_fc = nn.Linear(256 + 128, 128)
        self.fusion_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.residual = nn.Linear(256 + 128, 64)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
        text_dim = 256 * 3
        text_features = x[:, :text_dim]
        other_features = x[:, text_dim:]

        t1 = self.relu(self.text_bn1(self.text_fc1(text_features)))
        t2 = self.relu(self.text_bn2(self.text_fc2(t1)))
        t_attn = torch.sigmoid(self.text_attn(t2))
        t_out = t2 * t_attn

        o1 = self.relu(self.other_bn1(self.other_fc1(other_features)))
        o2 = self.relu(self.other_bn2(self.other_fc2(o1)))

        combined = torch.cat([t_out, o2], dim=1)
        f1 = self.relu(self.fusion_bn(self.fusion_fc(combined)))
        f2 = self.relu(self.bn2(self.fc2(f1)))
        residual = self.residual(combined)
        out = self.relu(f2 + residual)
        out = self.dropout(out)
        reg_out = self.regressor(out).squeeze(1)
        return reg_out


model = PlayCountPredictorV21(input_dim=X.shape[1]).to(device)
print(f"[train_v21_fallback.py] Model input dimension = {X.shape[1]}")

#############################################################
# 6) 训练准备
#############################################################
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15
)

best_val_loss = float('inf')
patience = 80
counter = 0
num_epochs = 400
val_loss_window = []

from torch.utils.data import DataLoader, WeightedRandomSampler

y_min = y_train_tensor.min().item()
y_max = y_train_tensor.max().item()
y_range = y_max - y_min
if y_range < 1e-6:
    sample_weights = torch.ones_like(y_train_tensor)
else:
    alpha = 5.0
    sample_weights = 1.0 + alpha * (y_train_tensor - y_min) / y_range

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

bar_format_style = ("{desc}|{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

#############################################################
# 7) 训练循环
#############################################################
for epoch in tqdm(range(num_epochs), desc="Training v21", bar_format=bar_format_style):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        noise = torch.randn_like(batch_X) * 0.02
        noisy_batch_X = batch_X + noise

        optimizer.zero_grad()
        pred_y = model(noisy_batch_X)

        real_play = torch.expm1(batch_y)
        weights = torch.ones_like(real_play)

        mask_1 = (real_play < 10)
        weights[mask_1] = 1.0

        mask_2 = (real_play >= 10) & (real_play < 100)
        weights[mask_2] = 1.3

        mask_3 = (real_play >= 100) & (real_play < 1000)
        weights[mask_3] = 1.8

        mask_4 = (real_play >= 1000) & (real_play < 1e4)
        weights[mask_4] = 1.2

        mask_5 = (real_play >= 1e4) & (real_play < 1e5)
        weights[mask_5] = 1.4

        mask_6 = (real_play >= 1e5) & (real_play < 1e6)
        weights[mask_6] = 1.8

        mask_7 = (real_play >= 1e6)
        weights[mask_7] = 2.5

        loss = criterion(pred_y, batch_y)
        loss = (loss * weights).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred_y = model(batch_X)
            vloss = criterion(pred_y, batch_y).mean()
            val_loss += vloss.item()

    avg_train = total_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    val_loss_window.append(avg_val)
    if len(val_loss_window) > 15:
        val_loss_window.pop(0)
    smoothed_val_loss = np.mean(val_loss_window)

    scheduler.step(smoothed_val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs} => TrainLoss={avg_train:.4f}, "
          f"ValLoss={avg_val:.4f}, Smoothed={smoothed_val_loss:.4f}, "
          f"LR={optimizer.param_groups[0]['lr']:.6f}")

    if smoothed_val_loss < best_val_loss:
        best_val_loss = smoothed_val_loss
        torch.save(model.state_dict(),
                   os.path.join(MODEL_DIR, 'play_count_model_v21.pth'))
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("[train_v21_fallback.py] Early stopping triggered.")
            break

print(f"[train_v21_fallback.py] Done. Best model saved to => {os.path.join(MODEL_DIR, 'play_count_model_v21.pth')}")

# === 额外评估 ===
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    state_dict = torch.load(os.path.join(MODEL_DIR, 'play_count_model_v21.pth'),
                            map_location=device)
    model.load_state_dict(state_dict)
except:
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'play_count_model_v21.pth'), map_location=device))

model.eval()

val_preds = []
val_targets = []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        pred_y = model(batch_X)
        val_preds.append(pred_y.cpu().numpy())
        val_targets.append(batch_y.cpu().numpy())

val_preds = np.concatenate(val_preds, axis=0)
val_targets = np.concatenate(val_targets, axis=0)

mse_val = mean_squared_error(val_targets, val_preds)
rmse_val = np.sqrt(mse_val)
mae_val = mean_absolute_error(val_targets, val_preds)

val_targets_exp = np.expm1(val_targets)
val_preds_exp = np.expm1(val_preds)

val_targets_for_mape = val_targets_exp.copy()
val_targets_for_mape[val_targets_for_mape < 1] = 1
mape_val = np.mean(np.abs((val_targets_exp - val_preds_exp) / val_targets_for_mape)) * 100

print("\n=== Final Validation Metrics on Best Model ===")
print(f"Validation RMSE (log-scale): {rmse_val:.4f}")
print(f"Validation MAE  (log-scale): {mae_val:.4f}")
print(f"Validation R²   (on original scale): {r2_score(val_targets_exp, val_preds_exp):.4f}")
print(f"Validation MAPE (on original scale): {mape_val:.2f}%")
