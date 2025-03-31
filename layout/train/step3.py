import csv
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

# API 配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEYS = [
    "sk-hdilnusolwkldxlyvzmjzaqkobspqdrorncuteywvqbuvlhw",  # 老的 API_KEY
    "sk-jagtlkdxrfrdiyoucgqqimndcvqfkrujvkvfjztevzgomojo"  # 新的 API_KEY
]
TREND_MODEL_ID = "Pro/deepseek-ai/DeepSeek-V3"


# 获取 Bilibili 热门视频标题
def fetch_hot_titles(max_retries=3, delay=2):
    url = "https://api.bilibili.com/x/web-interface/popular?Ps=100"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.bilibili.com/",
        "Origin": "https://www.bilibili.com"
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data["code"] == 0 and "list" in data["data"]:
                titles = [item["title"] for item in data["data"]["list"]]
                return titles
            else:
                print(f"❌ 获取 Bilibili 热点失败: {data.get('message', '未知错误')}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"❌ 获取 Bilibili 热点异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)  # 在重试前等待
            else:
                return []


# AI 趋势分析函数（单批处理）
def analyze_trending_relevance(titles, comments, hot_titles, api_key_idx):
    prompt = (
        "请分析以下10个视频标题和封面评论的趋势相关性，严格按照以下要求返回结果：\n"
        "1. **仅返回纯字符串，不包含任何 JSON 标记、多余文本、注释或说明**。\n"
        "2. 返回10组评分，每组对应一个标题和封面评论对，用 '|' 分隔每组评分。\n"
        "3. 每组评分包含4个字段，用空格分隔，顺序为：trending emotion visual creativity。\n"
        "4. 每个字段值为0到1之间的小数（保留2位小数）。\n"
        "   - trending: 时势热点（标题和封面是否紧跟当前热点，结合流行话题、事件或趋势，吸引关注）。\n"
        "   - emotion: 情感共鸣（标题和封面能否引起情感反应，如好奇、兴奋、感动、搞笑，促进点击）。\n"
        "   - visual: 视觉吸引力（封面设计是否吸引，能否在平台中脱颖而出）。\n"
        "   - creativity: 创意与独特性（标题和封面是否独特，能与其他内容区分，提供新鲜感）。\n"
        "5. 如果输入少于10个标题和评论对，仍需返回10组评分，未使用的组用 '0.00 0.00 0.00 0.00' 填充。\n"
        "6. 确保输出格式正确，所有数值在 0-1 范围内，无非法字符，请你在评价时带上更多的思考和客观性，让评分差距客观细致。\n"
        f"当前热点标题（参考用）：{', '.join(hot_titles[:30])}\n"
        f"输入数据：{str([{'title': t, 'comment': c} for t, c in zip(titles, comments)])}\n"
        "示例输出：\n"
        "0.86 0.77 0.97 0.83 | 0.65 0.67 0.71 0.64 | 0.35 0.25 0.47 0.92 | ...（共10组）"
    )
    payload = {
        "model": TREND_MODEL_ID,
        "stream": False,
        "max_tokens": 4096,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "frequency_penalty": 0.5,
        "n": 1,
        "stop": [],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {API_KEYS[api_key_idx % len(API_KEYS)]}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            try:
                groups = content.split("|")
                if len(groups) != 10:
                    raise ValueError(f"返回的评分组数不是10: {len(groups)}")
                result = []
                for group in groups:
                    values = group.strip().split()
                    if len(values) != 4:
                        raise ValueError(f"评分字段数不是4: {group}")
                    scores = [float(v) for v in values]
                    if not all(0 <= s <= 1 for s in scores):
                        raise ValueError(f"评分超出范围: {group}")
                    result.append(
                        {"trending": scores[0], "emotion": scores[1], "visual": scores[2], "creativity": scores[3]})
                return result
            except (ValueError, IndexError) as e:
                print(f"❌ 输出格式错误: {content}，错误: {e}")
                return [{"trending": 0.00, "emotion": 0.00, "visual": 0.00, "creativity": 0.00}] * 10
        else:
            print(f"❌ API 返回错误 [{response.status_code}]：{response.text}")
            return [{"trending": 0.00, "emotion": 0.00, "visual": 0.00, "creativity": 0.00}] * 10
    except Exception as e:
        print(f"❌ API 分析异常：{e}")
        return [{"trending": 0.00, "emotion": 0.00, "visual": 0.00, "creativity": 0.00}] * 10


# 处理单批数据（线程任务）
def process_batch(batch_data):
    df, start_idx, hot_titles, batch_idx = batch_data
    batch_titles = df["视频标题"][start_idx:start_idx + 10].tolist()
    batch_comments = df["封面评论"][start_idx:start_idx + 10].tolist()
    result = analyze_trending_relevance(batch_titles, batch_comments, hot_titles, batch_idx)
    return start_idx, result


# 重排序 CSV，异常数据移到末尾
def reorder_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    normal_data = df[
        (df["播放数"].fillna(0) != 0) &
        (~df["封面评论"].isna()) &
        (~df["封面评论"].str.contains("错误|分析失败", na=False))
        ]
    abnormal_data = df[
        (df["播放数"].fillna(0) == 0) |
        (df["封面评论"].isna()) |
        (df["封面评论"].str.contains("错误|分析失败", na=False))
        ]
    reordered_df = pd.concat([normal_data, abnormal_data], ignore_index=True)
    reordered_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")
    print(f"✅ CSV 已重排序，异常数据移至末尾：{csv_path}")


# 主逻辑
def process_csv(csv_path, original_csv_path):
    hot_titles = fetch_hot_titles()
    print(f"✅ 获取到 {len(hot_titles)} 个 Bilibili 热门视频标题")

    if not hot_titles:  # 如果没有获取到热点标题，退出程序
        print("❌ 未获取到热点标题，程序退出")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "视频标题" not in df.columns or "封面评论" not in df.columns:
        print("❌ CSV 缺少必要列：视频标题或封面评论")
        return

    # 初始化四列，确保数据类型为 float
    for col in ["trending", "emotion", "visual", "creativity"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

    total_rows = len(df)
    batch_size = 10
    completed_count = 0

    # 使用 64 线程并发处理
    with ThreadPoolExecutor(max_workers=64) as executor:
        batches = [(df, start_idx, hot_titles, i) for i, start_idx in enumerate(range(0, total_rows, batch_size))]
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}

        for future in future_to_batch:
            start_idx, result = future.result()
            print(f"\n📥 处理第 {start_idx + 1} 到 {min(start_idx + batch_size, total_rows)} 行")
            for i, json_result in enumerate(result):
                row_idx = start_idx + i
                if row_idx < total_rows:
                    df.at[row_idx, "trending"] = json_result["trending"]
                    df.at[row_idx, "emotion"] = json_result["emotion"]
                    df.at[row_idx, "visual"] = json_result["visual"]
                    df.at[row_idx, "creativity"] = json_result["creativity"]

            df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")
            print(f"💾 批次已保存到缓存 CSV: {csv_path}")

            completed_count += batch_size
            if completed_count % 100 == 0:
                shutil.copy(csv_path, original_csv_path)
                print(f"📝 已处理 {completed_count} 行，同步到原始文件: {original_csv_path}")

    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")
    shutil.copy(csv_path, original_csv_path)
    print(f"✅ 最终结果已保存到: {original_csv_path}")

    reorder_csv(original_csv_path)


if __name__ == "__main__":
    original_csv = r"C:\Users\ElmCose\Desktop\打印\temp\merged.csv"
    cache_csv = "cache_merged.csv"

    if os.path.exists(original_csv):
        shutil.copy(original_csv, cache_csv)
        print(f"🚀 开始处理 CSV（缓存文件：{cache_csv}）...")
        process_csv(cache_csv, original_csv)
        print("✅ 所有处理完成！")
    else:
        print(f"❌ 文件不存在：{original_csv}")
