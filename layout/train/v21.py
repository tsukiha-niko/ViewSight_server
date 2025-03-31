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

# Set random seed and device
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directory structure
TEMP_DIR = "./temp"
MODEL_DIR = "./models"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Data Loading and Preprocessing
df = pd.read_csv('merged.csv')
print(f"Loaded {len(df)} rows of data.")

scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[['粉丝数', 'p_rating', '时长（秒）']].values)  # Added 时长（秒）
joblib.dump(scaler, os.path.join(TEMP_DIR, 'scaler_v21.pkl'))

y = np.log1p(df['播放数'].values)
y_cls = (df['播放数'] > 100).astype(float).values

sentiments = np.array([SnowNLP(t).sentiments if isinstance(t, str) else 0.5 for t in df['视频标题']]).reshape(-1, 1)
sentiment_scaler = StandardScaler()
sentiments = sentiment_scaler.fit_transform(sentiments)
joblib.dump(sentiment_scaler, os.path.join(TEMP_DIR, 'sentiment_scaler_v21.pkl'))
X_numeric = np.hstack([X_numeric, sentiments, y_cls.reshape(-1, 1)])  # 5 features now

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
bert_model.eval()


def text_to_vector(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


def get_vectors(column, filename, desc):
    filepath = os.path.join(TEMP_DIR, filename)
    if os.path.exists(filepath):
        print(f"Loading cached BERT vectors for {column}...")
        vectors = np.load(filepath)
    else:
        print(f"Generating BERT vectors for {column}...")
        vectors = np.stack([text_to_vector(t) for t in tqdm(df[column], desc=desc)])
        np.save(filepath, vectors)
    return vectors


title_vectors = get_vectors('视频标题', 'title_vectors_v21.npy', "Titles")
cover_vectors = get_vectors('封面评论', 'cover_vectors_v21.npy', "Covers")
partition_vectors = get_vectors('分区', 'partition_vectors_v21.npy', "Partitions")


def get_pca(vectors, filename, n_components=256):  # Reduced from 384 to 256
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

X_text = np.hstack([title_pca, cover_pca, partition_pca])  # 256*3 = 768
X = np.hstack([X_text, X_numeric])

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


# Define Model (v21)
class PlayCountPredictorV21(nn.Module):
    def __init__(self, input_dim, text_dim=256 * 3):  # Adjusted text_dim
        super(PlayCountPredictorV21, self).__init__()
        self.text_fc1 = nn.Linear(text_dim, 512)  # Increased capacity
        self.text_bn1 = nn.BatchNorm1d(512)
        self.text_fc2 = nn.Linear(512, 256)  # Added layer
        self.text_bn2 = nn.BatchNorm1d(256)
        self.text_attn = nn.Linear(256, 256)

        self.other_fc1 = nn.Linear(input_dim - text_dim, 256)  # Increased capacity
        self.other_bn1 = nn.BatchNorm1d(256)
        self.other_fc2 = nn.Linear(256, 128)  # Added layer
        self.other_bn2 = nn.BatchNorm1d(128)

        self.fusion_fc = nn.Linear(256 + 128, 128)
        self.fusion_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.residual = nn.Linear(256 + 128, 64)
        self.dropout = nn.Dropout(0.4)  # Slightly increased dropout
        self.relu = nn.ReLU()
        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
        text_features = x[:, :256 * 3]
        other_features = x[:, 256 * 3:]

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
print(f"Revised Model (v21) input dimension: {X.shape[1]}")

# Training
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)  # Adjusted lr and weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

batch_size = 128  # Increased batch size
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

best_val_loss = float('inf')
patience = 80  # Reduced patience
counter = 0
num_epochs = 400
val_loss_window = []

for epoch in tqdm(range(num_epochs), desc="Training epochs"):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        noise = torch.randn_like(batch_X) * 0.02  # Increased noise
        batch_X = batch_X + noise
        optimizer.zero_grad()
        pred_y = model(batch_X)

        # Dynamic weights based on playback range
        weights = torch.ones_like(batch_y)
        weights[batch_y < np.log1p(100)] = 1.2
        weights[(batch_y >= np.log1p(100)) & (batch_y < np.log1p(100000))] = 1.5
        weights[(batch_y >= np.log1p(100000)) & (batch_y < np.log1p(1000000))] = 2.0
        weights[batch_y >= np.log1p(1000000)] = 2.5  # Increased weight for top tier

        loss = (criterion(pred_y, batch_y) * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred_y = model(batch_X)
            loss = criterion(pred_y, batch_y)
            val_loss += loss.item()

    avg_train = total_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    val_loss_window.append(avg_val)
    if len(val_loss_window) > 15:
        val_loss_window.pop(0)
    smoothed_val_loss = np.mean(val_loss_window)
    print(
        f"Epoch {epoch + 1}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}, Smoothed Val Loss = {smoothed_val_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

    scheduler.step(smoothed_val_loss)
    if smoothed_val_loss < best_val_loss:
        best_val_loss = smoothed_val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'play_count_model_v21.pth'))
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

print(f"Model saved to {os.path.join(MODEL_DIR, 'play_count_model_v21.pth')}")
