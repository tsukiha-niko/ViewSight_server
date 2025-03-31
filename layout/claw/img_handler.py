import configparser
import csv
import json
import logging
import time
from pathlib import Path

import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('relevancy_rating.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 读取配置文件
def load_config(config_filename="config.properties"):
    config = configparser.ConfigParser()
    try:
        config.read(config_filename, encoding='utf-8')
        api_key = config.get('api', 'api_key', fallback=None)
        input_csv = config.get('file', 'input_csv', fallback='bilibili_videos.csv')
        output_csv = config.get('file', 'output_csv', fallback='bilibili_videos_with_rating.csv')
        if not api_key:
            raise ValueError("配置文件中未找到 API key")
        logger.info("配置文件加载成功")
        return api_key, input_csv, output_csv
    except Exception as e:
        logger.error(f"读取配置文件失败: {e}")
        raise


# 调用 AI API 获取相关性评分
def get_relevancy_rating(title, cover_url, api_key):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 提示词设计：要求 AI 判断封面图片与标题的相关性，并返回 0-100 的评分
    prompt = (
        "请根据以下标题和图片封面，评估图片与标题的相关性。标题描述了视频内容，图片是视频的封面。返回一个 0-100 的整数评分，其中 0 表示完全不相关，100 表示高度相关。请时代热点简要说明你的评分依据。\n"
        f"标题: {title}\n"
        "图片: [见下方提供的图片链接]\n"
        "输出格式: {'rating': 评分, 'reason': '简要说明'}"
    )

    payload = {
        "model": "Qwen/Qwen2-VL-72B-Instruct",
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "stop": [],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "image_url": {
                            "detail": "auto",
                            "url": cover_url
                        },
                        "type": "image_url"
                    },
                    {
                        "text": prompt,
                        "type": "text"
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        result = response.json()

        # 提取评分和原因
        content = result['choices'][0]['message']['content']
        rating_data = json.loads(content)  # 期望 AI 返回 JSON 格式
        rating = int(rating_data['rating'])
        reason = rating_data.get('reason', '无说明')

        if not (0 <= rating <= 100):
            logger.warning(f"评分超出范围: {rating}，调整为边界值")
            rating = max(0, min(100, rating))

        logger.info(f"标题: {title} | 评分: {rating} | 原因: {reason}")
        return rating, reason
    except requests.exceptions.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        return None, f"API 请求错误: {e}"
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"解析 API 响应失败: {e} | 响应内容: {response.text}")
        return None, f"响应解析错误: {e}"


# 处理 CSV 文件并添加评分
def process_csv(api_key, input_csv, output_csv):
    # 读取输入 CSV
    rows = []
    with open(input_csv, 'r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames
        if 'cover_url' not in headers or 'title' not in headers:
            raise ValueError("输入 CSV 文件缺少 'cover_url' 或 'title' 列")
        rows = [row for row in reader]

    # 添加新列 'relevancy_rating' 到输出文件
    new_headers = headers + ['relevancy_rating']
    processed_count = 0
    save_interval = 10

    # 创建或覆盖输出文件
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=new_headers)
        writer.writeheader()

    # 逐行处理并追加写入
    with open(output_csv, 'a', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=new_headers)

        for i, row in enumerate(rows, 1):
            title = row['title']
            cover_url = row['cover_url']
            logger.info(f"处理第 {i}/{len(rows)} 行: {title}")

            rating, reason = get_relevancy_rating(title, cover_url, api_key)
            if rating is not None:
                row['relevancy_rating'] = rating
            else:
                row['relevancy_rating'] = ''  # 如果失败，留空
                logger.warning(f"第 {i} 行评分失败: {title}")

            writer.writerow(row)
            processed_count += 1

            # 每 10 条保存一次
            if processed_count % save_interval == 0:
                logger.info(f"已处理 {processed_count} 条数据，保存进度")
                csvfile.flush()  # 强制写入磁盘

            time.sleep(0.5)  # 避免 API 调用过于频繁

    logger.info(f"处理完成！共处理 {processed_count} 条数据，结果已保存到 {output_csv}")


# 主程序
def main():
    try:
        api_key, input_csv, output_csv = load_config()
        if not Path(input_csv).exists():
            raise FileNotFoundError(f"输入文件 {input_csv} 不存在")

        logger.info(f"开始处理文件: {input_csv}")
        process_csv(api_key, input_csv, output_csv)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
