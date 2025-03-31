import base64
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from io import BytesIO

import requests
from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CACHE_FILE_PATH = './cache/cache.json'


# 获取当前时间并删除一周前的缓存文件
def clean_img_cache():
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, 'r', encoding='utf-8') as cache_file:
                cache_data = json.load(cache_file)

            # 删除一周前的缓存记录
            expiration_time = datetime.now() - timedelta(weeks=1)
            for sha256_hash in list(cache_data.keys()):
                cached_time = datetime.strptime(cache_data[sha256_hash]['timestamp'], '%Y-%m-%d %H:%M:%S')
                if cached_time < expiration_time:
                    del cache_data[sha256_hash]

            # 更新缓存文件
            with open(CACHE_FILE_PATH, 'w', encoding='utf-8') as cache_file:
                json.dump(cache_data, cache_file, ensure_ascii=False, indent=4)
            logger.info("已清理过期缓存文件")

        except Exception as e:
            logger.error(f"清理缓存失败: {e}")


def download_image(image_url, save_path):
    """下载图片从URL或本地路径，并保存到本地"""
    if image_url.startswith('http'):
        # 从URL下载图片
        try:
            logger.info(f"从 URL 下载图片: {image_url}")
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
            logger.info(f"图片已保存到: {save_path}")
        except Exception as e:
            logger.error(f"下载图片失败: {e}")
            return False
    else:
        # 从本地路径复制图片
        try:
            shutil.copy(image_url, save_path)
            logger.info(f"本地图片已复制到: {save_path}")
        except Exception as e:
            logger.error(f"复制本地图片失败: {e}")
            return False
    return True


def image_to_base64(image_path):
    """将图片文件转换为base64字符串"""
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_str = base64.b64encode(image_data).decode("utf-8")
            return base64_str
    except Exception as e:
        logger.error(f"转换图片为 base64 失败: {e}")
        return None


def get_comment_util(image_url: str, config: dict) -> str:
    """
    分析视频封面图片并返回文本描述

    Args:
        image_url (str): 视频封面图片的URL或本地文件路径
        config (dict): 包含API端点和凭据的配置JSON

    Returns:
        str: 封面图片的文本描述或错误信息
    """
    # 提取图片API配置
    image_api_url = config.get("imageUrl", "")
    image_token = config.get("imageToken", "")
    image_model = config.get("imageModel", "")

    if not (image_api_url and image_token and image_model):
        logger.error("图片处理配置不完整")
        return "封面图片分析失败"

    # 为图片生成sha256哈希
    image_name = hashlib.sha256(image_url.encode('utf-8')).hexdigest()

    # 检查是否存在缓存结果
    cache_data = {}
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, 'r', encoding='utf-8') as cache_file:
                cache_data = json.load(cache_file)
        except Exception as e:
            logger.error(f"读取缓存文件失败: {e}")

    if image_name in cache_data:
        logger.info(f"使用缓存的分析结果: {image_name}")
        return cache_data[image_name].get("description", "封面图片分析失败")

    # 定义保存图片的目录
    img_cache_dir = './cache'
    if not os.path.exists(img_cache_dir):
        os.makedirs(img_cache_dir)

    image_path = os.path.join(img_cache_dir, image_name + ".jpg")

    # 如果图片未缓存，则下载
    if not os.path.exists(image_path):
        if not download_image(image_url, image_path):
            return "封面图片下载失败"

    # 将图片转换为base64
    base64_image = image_to_base64(image_path)
    if not base64_image:
        return "封面图片转码失败"

    # 构造图片分析的提示词
    prompt_text = (
        "请简要分析这张视频封面，回答以下问题：图中是否有标题文字？如果有，请说出是什么文字并告诉我他的大小样式；没有则说明无文字。"
        "然后极其的简明扼要的分析画面的构图、色彩风格、吸睛元素和整体感觉。控制在100字以内。不需要使用markdown和分段说明，只回答一个段落不要换行的纯文本即可"
    )

    payload = {
        "model": image_model,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
        "stop": []
    }

    headers = {
        "Authorization": f"Bearer {image_token}",
        "Content-Type": "application/json"
    }

    try:
        logger.info(f"调用图片处理API: {image_api_url}")
        response = requests.post(image_api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if "choices" not in result or not result["choices"]:
            logger.error("图片处理API返回格式无效")
            return "封面图片分析失败"
        content = result["choices"][0]["message"].get("content", "").strip()

        # 缓存结果
        cache_data[image_name] = {
            "description": content,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 将缓存保存回文件
        with open(CACHE_FILE_PATH, 'w', encoding='utf-8') as cache_file:
            json.dump(cache_data, cache_file, ensure_ascii=False, indent=4)

        return content if content else "封面图片分析失败"
    except Exception as e:
        logger.error(f"调用图片处理API异常: {e}")
        return "封面图片分析失败"


def get_video_summary_util(config: dict, frame_analysis: list, prompt: str) -> str:
    """
    根据帧分析结果生成视频总结

    Args:
        config (dict): 包含文本API端点和凭据的配置JSON
        frame_analysis (list): 帧分析结果列表，每个元素包含frame、timestamp、subtitle、analysis
        prompt (str): 用于生成总结的提示词

    Returns:
        str: 视频总结文本或错误信息
    """
    # 提取文本API配置
    text_api_url = config.get("textUrl", "")
    text_token = config.get("textToken", "")
    text_model = config.get("textModel", "")

    if not (text_api_url and text_token and text_model):
        logger.error("文本处理配置不完整")
        return "视频总结生成失败"

    # 为帧分析生成唯一标识（基于提示词和帧分析内容的哈希）
    analysis_str = json.dumps(frame_analysis, ensure_ascii=False) + prompt
    summary_key = hashlib.sha256(analysis_str.encode('utf-8')).hexdigest()

    # 检查缓存
    cache_data = {}
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, 'r', encoding='utf-8') as cache_file:
                cache_data = json.load(cache_file)
        except Exception as e:
            logger.error(f"读取缓存文件失败: {e}")

    if summary_key in cache_data:
        logger.info(f"使用缓存的视频总结: {summary_key}")
        return cache_data[summary_key].get("description", "视频总结生成失败")

    # 构造文本API请求
    payload = {
        "model": text_model,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stop": []
    }

    headers = {
        "Authorization": f"Bearer {text_token}",
        "Content-Type": "application/json"
    }

    try:
        logger.info(f"调用文本处理API生成视频总结: {text_api_url}")
        response = requests.post(text_api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if "choices" not in result or not result["choices"]:
            logger.error("文本处理API返回格式无效")
            return "视频总结生成失败"
        content = result["choices"][0]["message"].get("content", "").strip()

        # 缓存结果
        cache_data[summary_key] = {
            "description": content,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 将缓存保存回文件
        with open(CACHE_FILE_PATH, 'w', encoding='utf-8') as cache_file:
            json.dump(cache_data, cache_file, ensure_ascii=False, indent=4)

        return content if content else "视频总结生成失败"
    except Exception as e:
        logger.error(f"调用文本处理API异常: {e}")
        return "视频总结生成失败"


if __name__ == "__main__":
    # 测试用
    pass
