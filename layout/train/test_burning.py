import argparse
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from snownlp import SnowNLP
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cuda.matmul.allow_tf32 = True

# =========== 1. 加载预处理对象 ===========
scaler = joblib.load('./temp/scaler_v21.pkl')
pca_title = joblib.load('./temp/pca_title_v21.pkl')
pca_cover = joblib.load('./temp/pca_cover_v21.pkl')
pca_partition = joblib.load('./temp/pca_partition_v21.pkl')
sentiment_scaler = joblib.load('./temp/sentiment_scaler_v21.pkl')

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
bert_model.eval()


def text_to_vector(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


class PlayCountPredictorV21(nn.Module):
    def __init__(self, input_dim, text_dim=256 * 3):
        super(PlayCountPredictorV21, self).__init__()
        self.text_fc1 = nn.Linear(text_dim, 512)
        self.text_bn1 = nn.BatchNorm1d(512)
        self.text_fc2 = nn.Linear(512, 256)
        self.text_bn2 = nn.BatchNorm1d(256)
        self.text_attn = nn.Linear(256, 256)

        self.other_fc1 = nn.Linear(input_dim - text_dim, 256)
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


# =========== 2. 数据过滤 & 读取 ===========
df_test = pd.read_csv(R'C:\Users\ElmCose\Desktop\打印\temp\merged.csv')
df_test = df_test[df_test['播放数'] > 0].copy()
print(f"After filtering 0-play, remaining {len(df_test)} rows.")
df_test = df_test.drop_duplicates(subset='视频标题', keep='first')
print(f"After removing duplicates, remaining {len(df_test)} rows.")

sample_size = min(100, len(df_test))
if sample_size < 100:
    print(f"Warning: Only {sample_size} samples available, less than requested 100.")
df_sample = df_test.sample(n=sample_size, random_state=42).reset_index(drop=True)

print("Generating BERT vectors for sample texts...")
title_vecs = np.stack([text_to_vector(t) for t in tqdm(df_sample['视频标题'], desc="Titles")])
cover_vecs = np.stack([text_to_vector(c) for c in tqdm(df_sample['封面评论'], desc="Covers")])
part_vecs = np.stack([text_to_vector(p) for p in tqdm(df_sample['分区'], desc="Partitions")])

title_pca = pca_title.transform(title_vecs)
cover_pca = pca_cover.transform(cover_vecs)
part_pca = pca_partition.transform(part_vecs)

X_text = np.hstack([title_pca, cover_pca, part_pca])

sentiments = np.array([SnowNLP(t).sentiments for t in df_sample['视频标题']]).reshape(-1, 1)
sentiments = sentiment_scaler.transform(sentiments)
y_cls = (df_sample['播放数'] > 100).astype(float).values.reshape(-1, 1)

numeric_cols = ['粉丝数', 'p_rating', '时长（秒）', 'trending', 'emotion', 'visual', 'creativity']
X_numeric_scaled = scaler.transform(df_sample[numeric_cols].fillna(0).values)
X_numeric = np.hstack([X_numeric_scaled, sentiments, y_cls])

input_dim = X_text.shape[1] + X_numeric.shape[1]
X_sample = np.hstack([X_text, X_numeric])

# =========== 3. 加载模型 ===========
model = PlayCountPredictorV21(input_dim=input_dim).to(device)
try:
    state_dict = torch.load('./models/play_count_model_v21.pth', map_location=device)
    model.load_state_dict(state_dict)
except:
    model.load_state_dict(torch.load('./models/play_count_model_v21.pth', map_location=device))
model.eval()

# =========== 4. 推理 ===========
X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)
with torch.no_grad():
    pred_reg = model(X_tensor).cpu().numpy()

pred_play_count = np.expm1(pred_reg)


# =========== 5. 后处理 => 以 pred_pp 为准 ===========
def final_post_process(val):
    if val < 10:
        return val * 1.2
    elif val < 100:
        return val * 0.85
    elif val < 1000:
        return val * 0.45
    elif val < 1e4:
        return val * 1.1
    elif val < 1e5:
        return val * 1.2
    elif val < 1e6:
        return val * 1.3
    else:
        return val * 1.5


pred_pp = np.array([final_post_process(x) for x in pred_play_count])
max_val = df_test['播放数'].max() * 2
pred_pp = np.clip(pred_pp, 0, max_val)

y_actual = df_sample['播放数'].values

# =========== 6. 评估 ===========
rmse = np.sqrt(mean_squared_error(y_actual, pred_pp))
mae = mean_absolute_error(y_actual, pred_pp)
y_mape = y_actual.astype(float).copy()
y_mape[y_mape < 1] = 1
mape = np.mean(np.abs((y_actual - pred_pp) / y_mape)) * 100
r2 = r2_score(y_actual, pred_pp)
med_ae = np.median(np.abs(y_actual - pred_pp))

print("\n=== 预测结果表格 ===")
print(f"{'视频标题':<50} {'实际播放量':>12} {'预测播放量':>12}")
print("-" * 74)
for i in range(len(df_sample)):
    title = df_sample['视频标题'][i]
    if len(title) > 48:
        title = title[:48] + "..."
    print(f"{title:<50} {int(y_actual[i]):>12} {int(pred_pp[i]):>12}")

print("\n=== 模型评估指标 ===")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Median Absolute Error: {med_ae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R² Score: {r2:.4f}")

df_compare = pd.DataFrame({
    '视频标题': df_sample['视频标题'],
    '实际播放数': y_actual,
    '预测播放数': pred_pp
})
df_compare['误差'] = df_compare['实际播放数'] - df_compare['预测播放数']
df_compare['绝对误差'] = df_compare['误差'].abs()

threshold = 1_000_000
df_baokuan = df_compare[df_compare['实际播放数'] >= threshold].copy()
if len(df_baokuan) > 0:
    print("\n=== 爆款视频(播放量>=100万)的单独评估 ===")
    bk_rmse = np.sqrt(mean_squared_error(df_baokuan['实际播放数'], df_baokuan['预测播放数']))
    bk_mae = mean_absolute_error(df_baokuan['实际播放数'], df_baokuan['预测播放数'])
    bk_med_ae = np.median(np.abs(df_baokuan['实际播放数'] - df_baokuan['预测播放数']))
    arr_act = df_baokuan['实际播放数'].values.copy()
    arr_act[arr_act < 1] = 1
    bk_mape = np.mean(np.abs((df_baokuan['实际播放数'] - df_baokuan['预测播放数']) / arr_act)) * 100
    print(f"爆款视频数量: {len(df_baokuan)}")
    print(f"爆款RMSE: {bk_rmse:.2f}")
    print(f"爆款MAE: {bk_mae:.2f}")
    print(f"爆款MedianAE: {bk_med_ae:.2f}")
    print(f"爆款MAPE: {bk_mape:.2f}%")

print("\n=== 误差分析：绝对误差最大的前10个样本 ===")
print(df_compare.sort_values('绝对误差', ascending=False).head(10))

# =========== 7. 作图 ===========
df_plot = df_compare.copy().reset_index(drop=True)


def is_big_err(row):
    a = row['实际播放数']
    p = row['预测播放数']
    if a < 1 or p < 1:
        return False
    ratio = p / a
    return abs(math.log10(ratio)) > 1


big_err = df_plot.apply(is_big_err, axis=1)

fig1 = plt.figure(figsize=(15, 8))
indices = np.arange(len(df_plot))
width = 0.35

plt.bar(indices - width / 2, df_plot['预测播放数'], width=width, color='skyblue', label='预测播放量', align='center')
plt.bar(indices + width / 2, df_plot['实际播放数'], width=width, color='salmon', label='实际播放量', align='center')

plt.yscale('log')
plt.xlabel('视频样本序号（标红为误差>1个数量级）')
plt.ylabel('播放量 (对数刻度)')
plt.title('预测播放量与实际播放量柱状对比图 (v21)')

plt.xticks(indices, [str(i) for i in indices])
for label, is_red in zip(plt.gca().get_xticklabels(), big_err):
    label.set_color('red' if is_red else 'black')

plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('bar_comparison_v21.png')

fig2 = plt.figure(figsize=(10, 8))
colors = []
count_within_half_ord = 0
count_within_1ord = 0
for i, row in df_plot.iterrows():
    a = row['实际播放数']
    p = row['预测播放数']
    if a < 1 or p < 1:
        colors.append('green')  # 小于 1 的默认绿色
        continue
    ratio = p / a
    log_ratio = abs(math.log10(ratio))
    if log_ratio <= 0.5:
        colors.append('green')
        count_within_half_ord += 1
        count_within_1ord += 1
    elif log_ratio <= 1:
        colors.append('blue')
        count_within_1ord += 1
    else:
        colors.append('red')

within_half_percent = count_within_half_ord / len(df_plot) * 100
within_1ord_percent = count_within_1ord / len(df_plot) * 100

plt.scatter(df_plot['实际播放数'], df_plot['预测播放数'], c=colors, alpha=0.7)
max_val = max(df_plot['实际播放数'].max(), df_plot['预测播放数'].max())
plt.plot([1, max_val], [1, max_val], color='black', linestyle='--', label='理想拟合')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='误差≤0.5个数量级', markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='误差0.5-1个数量级', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='误差>1个数量级', markerfacecolor='red', markersize=10),
    Line2D([0], [0], color='black', linestyle='--', label='理想拟合')
]
plt.legend(handles=legend_elements)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('实际播放量')
plt.ylabel('预测播放量')
plt.title(f'散点图（对数刻度）(v21), {within_half_percent:.2f}% 误差≤0.5数量级, {within_1ord_percent:.2f}% 误差≤1数量级')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('scatter_pred_vs_actual_v21.png')

fig3 = plt.figure(figsize=(10, 6))
residuals = df_plot['实际播放数'] - df_plot['预测播放数']
plt.hist(residuals, bins=20, color='purple', alpha=0.7)
plt.xlabel('残差 (实际 - 预测)')
plt.ylabel('样本数')
plt.title('预测残差分布直方图 (v21)')
plt.grid(True, ls="--")
plt.tight_layout()
plt.savefig('residual_histogram_v21.png')

plt.show()
print("所有图表已保存 => bar_comparison_v21.png, scatter_pred_vs_actual_v21.png, residual_histogram_v21.png")

# =========== 8. Flask & API ===========
app = Flask(__name__)


def final_post_process_api(val):
    if val < 10:
        return val * 1.2
    elif val < 100:
        return val * 0.85
    elif val < 1000:
        return val * 0.45
    elif val < 1e4:
        return val * 1.1
    elif val < 1e5:
        return val * 1.2
    elif val < 1e6:
        return val * 1.3
    else:
        return val * 1.5


@app.route('/predict', methods=['GET'])
def predict():
    video_title = request.args.get('视频标题', default='', type=str)
    cover_comment = request.args.get('封面评论', default='', type=str)
    partition = request.args.get('分区', default='', type=str)
    fans_count = request.args.get('粉丝数', default=0, type=float)
    p_rating = request.args.get('p_rating', default=0, type=float)
    duration = request.args.get('时长（秒）', default=0, type=float)
    trending = request.args.get('trending', default=0.5, type=float)
    emotion = request.args.get('emotion', default=0.5, type=float)
    visual = request.args.get('visual', default=0.5, type=float)
    creativity = request.args.get('creativity', default=0.5, type=float)

    if not video_title or not cover_comment or not partition:
        return jsonify({"error": "缺少文本字段"}), 400

    title_vec = text_to_vector(video_title)
    cover_vec = text_to_vector(cover_comment)
    part_vec = text_to_vector(partition)
    t_pca = pca_title.transform(title_vec.reshape(1, -1))
    c_pca = pca_cover.transform(cover_vec.reshape(1, -1))
    p_pca = pca_partition.transform(part_vec.reshape(1, -1))
    X_text_instance = np.hstack([t_pca, c_pca, p_pca])

    sentiment_val = SnowNLP(video_title).sentiments
    sentiment_scaled = sentiment_scaler.transform([[sentiment_val]])
    y_cls_val = np.array([[1.0 if fans_count > 100 else 0.0]])
    numeric_features = np.array([[fans_count, p_rating, duration, trending, emotion, visual, creativity]])
    numeric_scaled = scaler.transform(numeric_features)
    X_numeric_instance = np.hstack([numeric_scaled, sentiment_scaled, y_cls_val])

    X_instance = np.hstack([X_text_instance, X_numeric_instance])
    X_tensor = torch.tensor(X_instance, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_log = model(X_tensor).cpu().numpy()[0]
    raw_prediction = np.expm1(pred_log)

    pp = final_post_process_api(raw_prediction)
    max_val2 = df_test['播放数'].max() * 2
    pp = np.clip(pp, 0, max_val2)

    return jsonify({
        "predicted_play_count": float(pp),
        "raw_regression": float(pred_log),
        "post_processed": float(pp)
    })


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"error": "需要提供数组形式的数据"}), 400

    results = []
    for record in data:
        video_title = record.get('视频标题', '')
        cover_comment = record.get('封面评论', '')
        partition = record.get('分区', '')
        fans_count = record.get('粉丝数', 0)
        p_rating = record.get('p_rating', 0)
        duration = record.get('时长（秒）', 0)
        trending = record.get('trending', 0.5)
        emotion = record.get('emotion', 0.5)
        visual = record.get('visual', 0.5)
        creativity = record.get('creativity', 0.5)

        title_vec = text_to_vector(video_title)
        cover_vec = text_to_vector(cover_comment)
        part_vec = text_to_vector(partition)

        t_pca = pca_title.transform(title_vec.reshape(1, -1))
        c_pca = pca_cover.transform(cover_vec.reshape(1, -1))
        p_pca = pca_partition.transform(part_vec.reshape(1, -1))
        X_text_instance = np.hstack([t_pca, c_pca, p_pca])

        sentiment_val = SnowNLP(video_title).sentiments
        sentiment_scaled = sentiment_scaler.transform([[sentiment_val]])
        y_cls_val = np.array([[1.0 if fans_count > 100 else 0.0]])
        numeric_features = np.array([[fans_count, p_rating, duration, trending, emotion, visual, creativity]])
        numeric_scaled = scaler.transform(numeric_features)
        X_numeric_instance = np.hstack([numeric_scaled, sentiment_scaled, y_cls_val])

        X_instance = np.hstack([X_text_instance, X_numeric_instance])
        X_tensor = torch.tensor(X_instance, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_log = model(X_tensor).cpu().numpy()[0]
        raw_pred = np.expm1(pred_log)

        pp = final_post_process_api(raw_pred)
        max_val2 = df_test['播放数'].max() * 2
        pp = np.clip(pp, 0, max_val2)

        results.append({
            "视频标题": video_title,
            "predicted_play_count": float(pp),
            "raw_regression": float(pred_log),
            "post_processed": float(pp)
        })

    return jsonify(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['eval', 'server', 'all'], default='server',
                        help='eval: 只评估, server: 只启服务, all:都执行')
    args = parser.parse_args()

    if args.mode in ['eval', 'all']:
        print("开始评估与绘图...")
    if args.mode in ['server', 'all']:
        print("启动 Web 服务, 端口10650...")
        app.run(host='0.0.0.0', port=10650)
