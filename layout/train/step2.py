import base64
import concurrent.futures
import csv
import io
import os
import shutil

import pandas as pd
import requests
from PIL import Image

# ====================== API 配置 ======================
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEYS = [
    "sk-hdilnusolwkldxlyvzmjzaqkobspqdrorncuteywvqbuvlhw",  # 老的 API_KEY
    "sk-jagtlkdxrfrdiyoucgqqimndcvqfkrujvkvfjztevzgomojo"  # 新的 API_KEY
]
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"


# ====================== 图像处理函数 ======================
def download_image(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"❌ 下载失败：{url}，错误：{e}")
        return None


def prepare_image_for_model(image, target_size_kb=10, max_resolution=(256, 256), color_reduction=64):
    def resize_image(image, max_size):
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image

    def reduce_colors(image, colors):
        return image.convert("P", palette=Image.ADAPTIVE, colors=colors).convert("RGB")

    def compress_webp(image, target_size_kb, min_quality=10):
        best_image = None
        best_size = float('inf')
        for quality in range(95, min_quality - 1, -5):
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP", quality=quality)
            size_kb = buffer.tell() / 1024
            if size_kb <= target_size_kb:
                buffer.seek(0)
                return Image.open(buffer).convert("RGB")
            if size_kb < best_size:
                best_size = size_kb
                best_image = buffer
        print(f"⚠️ 无法压缩到 {target_size_kb}KB，已选用最小版本（{best_size:.2f}KB）")
        best_image.seek(0)
        return Image.open(best_image).convert("RGB")

    image = resize_image(image, max_resolution)
    image = reduce_colors(image, color_reduction)
    image = compress_webp(image, target_size_kb)
    return image


# ====================== AI 分析函数（调用 API） ======================
def analyze_image(image, api_key_idx):
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_data_uri = f"data:image/jpeg;base64,{img_b64}"
        prompt = (
            "请简要分析这张视频封面，回答以下问题：图中是否有标题文字？如果有，请说出是什么文字并告诉我他的大小样式；没有则说明无文字。"
            "然后极其的简明扼要的分析画面的构图、色彩风格、吸睛元素和整体感觉。控制在100字以内。不需要使用markdown和分段说明，只回答一段段落纯文本即可"
        )
        payload = {
            "model": MODEL_ID,
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "stop": [],
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": prompt}
                ]}
            ]
        }
        headers = {
            "Authorization": f"Bearer {API_KEYS[api_key_idx % len(API_KEYS)]}",  # 交替使用 API_KEY
            "Content-Type": "application/json"
        }
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            print(f"❌ API 返回错误 [{response.status_code}]：{response.text}")
            return "分析失败"
        result = response.json()
        reply = result["choices"][0]["message"]["content"].strip()
        return reply if reply else "分析失败"
    except Exception as e:
        print(f"❌ API 分析异常：{e}")
        return "分析失败"


# ====================== 单行处理函数 ======================
def process_row(idx, cover_url, output_folder):
    print(f"\n📥 处理第 {idx + 1} 行，封面链接：{cover_url}")
    image = download_image(cover_url)
    if image is None:
        return idx, "下载失败"
    image = prepare_image_for_model(image)
    image_path = os.path.join(output_folder, f"cover_{idx + 1}.jpg")
    try:
        image.save(image_path)
    except Exception as e:
        print(f"❌ 保存图片失败：{image_path}，错误：{e}")
    print("🧠 正在分析...")
    # 使用 idx 对 API_KEYS 取模，交替分配
    analysis = analyze_image(image, idx)
    if analysis.lower() in ["分析失败", ""]:
        print("⚠️ 分析失败，保留空白以便下次重试。")
    else:
        print(f"✅ 输出：{analysis}")
    return idx, analysis


# ====================== 主逻辑 ======================
def process_csv(csv_path, original_csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "封面评论" not in df.columns:
        df["封面评论"] = pd.NA

    output_folder = "compressed_images"
    os.makedirs(output_folder, exist_ok=True)

    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:  # 提升至 20 线程
        future_to_idx = {}
        for idx, row in df.iterrows():
            cover_url = row.get("封面")
            current_comment = row.get("封面评论")

            if pd.isna(cover_url):
                continue
            if not (pd.isna(current_comment) or str(current_comment).strip() in ["", "下载失败", "分析失败"]):
                continue

            future = executor.submit(process_row, idx, cover_url, output_folder)
            future_to_idx[future] = idx

        for future in concurrent.futures.as_completed(future_to_idx):
            idx, result = future.result()
            df.at[idx, "封面评论"] = result

            # 每次处理完一行，保存到 cache_merged.csv
            if completed_count % 10 == 0:
                try:
                    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")
                    print(f"💾 第 {idx + 1} 行已写入缓存 CSV: {csv_path}")
                except Exception as e:
                    print(f"❌ 写入缓存 CSV 失败: {e}")

            completed_count += 1
            if completed_count % 100 == 0:
                # 每 100 次尝试同步到 merged.csv
                try:
                    shutil.copy(csv_path, original_csv_path)
                    print(f"📝 已处理 {completed_count} 条，原始文件同步更新: {original_csv_path}")
                except Exception as e:
                    print(f"❌ 同步到原始文件失败: {e}，保留缓存文件: {csv_path}")

    try:
        df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")
        df.to_csv(original_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")
        print(f"✅ 最终结果已保存到缓存文件: {csv_path}")
    except Exception as e:
        print(f"❌ 最终保存缓存文件失败: {e}")


if __name__ == "__main__":
    original_csv = r"C:\Users\ElmCose\Desktop\打印\temp\merged.csv"
    cache_csv = "cache_merged.csv"

    # 如果 merged.csv 存在，复制到 cache_merged.csv 作为初始工作文件
    if os.path.exists(original_csv):
        try:
            shutil.copy(original_csv, cache_csv)
            print(f"🚀 开始处理 CSV（缓存文件：{cache_csv}）...")
            process_csv(cache_csv, original_csv)
            print("✅ 所有处理完成！")
            print(f"📌 最终结果保存在: {original_csv}")
        except Exception as e:
            print(f"❌ 初始化或处理失败: {e}")
    else:
        print(f"❌ 文件不存在：{original_csv}")
