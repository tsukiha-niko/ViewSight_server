import logging
import math
import re

import requests
from flask import request, jsonify

from config_util import get_config_from_request
from utils.get_bewrite_util import get_bewrite_util
from utils.get_comment_util import get_comment_util
from utils.get_model_predict_util import call_predict_api

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("blink_handler.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def is_valid_bv_av(video_input):
    """Check if the input is a valid BV or AV number"""
    bv_pattern = r"^BV[0-9A-Za-z]{8,10}$"  # BV号: BV + 8-10位字母数字
    av_pattern = r"^AV[0-9]+$"  # AV号: AV + 纯数字
    if re.match(bv_pattern, video_input):
        return "BV", video_input
    if re.match(av_pattern, video_input):
        return "AV", video_input[2:]  # 去掉 "AV" 前缀
    return None, None


def extract_video_id(video_input):
    """Extract or validate video ID from URL or direct BV/AV input"""
    # 先检查是否直接是 BV 或 AV 号
    video_type, video_id = is_valid_bv_av(video_input)
    if video_type:
        return video_id

    # 如果不是直接 BV/AV，则尝试从 URL 中提取
    url_pattern = r"/video/([^/?]+)"
    match = re.search(url_pattern, video_input)
    return match.group(1) if match else None


def get_video_info(video_id, cookie):
    """Fetch video information from Bilibili API"""
    api_url = "https://api.bilibili.com/x/web-interface/view"
    params = {"bvid": video_id} if video_id.lower().startswith("bv") else {"aid": video_id}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com/"}
    cookies = {k.strip(): v.strip() for part in cookie.split(";") if "=" in part for k, v in [part.split("=", 1)]}

    try:
        response = requests.get(api_url, params=params, headers=headers, cookies=cookies, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            logger.error(f"视频详情接口错误: {data.get('message')}")
            return None
        video_data = data["data"]
        return {
            "视频标题": video_data.get("title", ""),
            "封面评论": video_data.get("pic", ""),
            "分区": video_data.get("tname", ""),
            "owner_mid": video_data.get("owner", {}).get("mid", ""),
            "时长（秒）": video_data.get("duration", 0)
        }
    except Exception as e:
        logger.error(f"调用视频详情接口异常: {e}")
        return None


def get_up_info(mid):
    """Fetch UP主的 information from Bilibili API"""
    api_url = f"https://api.bilibili.com/x/web-interface/card?mid={mid}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com/"}

    try:
        response = requests.get(api_url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            logger.error(f"UP主信息接口错误: {data.get('message')}")
            return None
        card_data = data["data"]
        return {
            "粉丝数": card_data.get("follower", 0),
            "up历史点赞数": card_data.get("like_num", 0),
            "up历史稿件数": card_data.get("archive_count", 0)
        }
    except Exception as e:
        logger.error(f"调用 UP 主信息接口异常: {e}")
        return None


def handle_blink():
    """
    Handle /analyze_video_link POST request:
    1. Extract video link or BV/AV number and config
    2. Get video info (title, cover URL, etc.)
    3. Use get_comment_util to analyze cover image
    4. Use get_bewrite_util to get trending scores
    5. Get UP主 info and calculate p_rating
    6. Call predict API with all data including trending scores
    7. Return results with range_score
    """
    config = get_config_from_request()
    if not config:
        return jsonify({"error": "配置未提供"}), 400

    cookie = config.get("cookie", "")
    aiserverurl = config.get("aiServerUrl", "http://192.168.1.3:10650")

    req_data = request.get_json()
    if not req_data or "videoLink" not in req_data:
        return jsonify({"error": "缺少参数: videoLink"}), 400

    video_input = req_data["videoLink"].strip()
    logger.info(f"接收到的输入: {video_input}")
    print(f"接收到的输入: {video_input}")

    video_id = extract_video_id(video_input)
    if not video_id:
        logger.error("无法提取或验证视频ID")
        return jsonify({"error": "无效的视频链接或BV/AV号"}), 400

    logger.info(f"提取到的视频ID: {video_id}")
    print(f"提取到的视频ID: {video_id}")

    # Step 1: Get video details
    video_info = get_video_info(video_id, cookie)
    if not video_info:
        return jsonify({"error": "获取视频信息失败"}), 500

    owner_mid = video_info.get("owner_mid")
    if not owner_mid:
        logger.error("视频信息中缺少 up 主 ID")
        return jsonify({"error": "视频信息中缺少 up 主 ID"}), 500

    # Step 2: Analyze cover image using get_comment_util
    cover_image_url = video_info.get("封面评论", "")
    cover_comment = get_comment_util(cover_image_url, config)
    logger.info(f"封面分析结果: {cover_comment}")

    # Step 3: Get trending scores using get_bewrite_util
    video_title = video_info.get("视频标题", "")
    trending, emotion, visual, creativity = get_bewrite_util(video_title, cover_comment, config)
    logger.info(
        f"趋势分析结果: trending={trending:.2f}, emotion={emotion:.2f}, visual={visual:.2f}, creativity={creativity:.2f}")

    # Step 4: Get UP主 info
    up_info = get_up_info(owner_mid)
    if not up_info:
        return jsonify({"error": "获取 UP 主信息失败"}), 500

    archive_count = up_info.get("up历史稿件数", 0)
    like_num = up_info.get("up历史点赞数", 0)
    p_rating = like_num / archive_count if archive_count > 0 else 0

    # Step 5: Duration handling
    duration = video_info.get("时长（秒）", 0)
    if not isinstance(duration, int):
        try:
            duration = int(duration)
        except:
            duration = 0

    # Step 6: Assemble data and call predict API with trending scores
    combined_info = {
        "视频标题": video_title,
        "封面评论": cover_comment,
        "分区": video_info.get("分区", ""),
        "粉丝数": up_info.get("粉丝数", 0),
        "up历史点赞数": like_num,
        "up历史稿件数": archive_count,
        "p_rating": p_rating,
        "时长（秒）": duration
    }

    logger.info("视频数据如下：")
    for key, value in combined_info.items():
        logger.info(f"{key}: {value}")
        print(f"{key}: {value}")

    predict_result = call_predict_api(
        aiserverurl,
        combined_info["视频标题"],
        combined_info["封面评论"],
        combined_info["分区"],
        combined_info["粉丝数"],
        combined_info["p_rating"],
        combined_info["时长（秒）"],
        trending,
        emotion,
        visual,
        creativity
    )
    if not predict_result:
        return jsonify({"error": "调用预测接口失败"}), 500

    # Step 7: Calculate range_score
    predicted_play_count = predict_result.get("predicted_play_count")
    predicted_play_count_0_5 = None
    range_score = "N/A"
    if predicted_play_count is not None:
        try:
            pred = float(predicted_play_count)
            lower_bound = int(round(10 ** (math.log10(pred) - 1)))
            upper_bound = int(round(10 ** (math.log10(pred) + 1)))
            range_score = f"{lower_bound} - {upper_bound}"

            # Calculate the predicted_play_count_0.5
            predicted_play_count_0_5 = (
                f"{int(round(pred * 10 ** -0.5))} - {int(round(pred * 10 ** 0.5))}"
            )
        except Exception as e:
            logger.error(f"计算 range_score 异常: {e}")

    # Step 8: Prepare response with additional trending scores
    response_data = {
        "step": "链接解析",
        "message": "✅ 基础解析成功\n视频分析请移步专业版",
        "predicted_play_count": predicted_play_count,
        "range_score": range_score,
        "estimated7DayViews": predicted_play_count_0_5,
        "raw_regression": predict_result.get("raw_regression"),
        "post_processed": predict_result.get("post_processed"),
        "trending_analysis": {
            "trending": f"{trending:.2f}",
            "emotion": f"{emotion:.2f}",
            "visual": f"{visual:.2f}",
            "creativity": f"{creativity:.2f}"
        }
    }

    logger.info(f"返回前端数据: {response_data}")
    print(f"返回前端数据: {response_data}")

    return jsonify(response_data)


if __name__ == "__main__":
    # For testing purposes
    pass
