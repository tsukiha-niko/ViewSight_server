import logging

from flask import request, jsonify

from config_util import get_config_from_request
from utils.get_bewrite_util import get_bewrite_util
from utils.get_model_predict_util import call_predict_api  # Import the existing utility

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("bvideo_handler.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def handle_bvideo():
    """
    Handle /analyze_video_cover POST request:
    1. Extract config and combined_info from request
    2. Use get_bewrite_util to get trending scores based on title and cover comment
    3. Return results with trending, emotion, visual, and creativity scores
    """
    # Step 1: Extract config and combined_info
    config = get_config_from_request()
    if not config:
        return jsonify({"error": "配置未提供"}), 400

    req_data = request.get_json()
    if not req_data or "combined_info" not in req_data:
        return jsonify({"error": "缺少参数: combined_info"}), 400

    combined_info = req_data.get("combined_info", {})
    video_title = combined_info.get("视频标题", "")
    cover_comment = combined_info.get("封面评论", "")
    script_content = combined_info.get("视频文案", "")

    # Validate required fields
    if not video_title or not cover_comment:
        logger.error("combined_info 中缺少必要的字段: 视频标题 或 封面评论")
        return jsonify({"error": "combined_info 中缺少视频标题或封面评论"}), 400

    logger.info(f"接收到的视频标题: {video_title}")
    logger.info(f"接收到的封面评论: {cover_comment}")
    logger.info(f"接收到的视频文案: {script_content}")
    print(f"接收到的视频标题: {video_title}")
    print(f"接收到的封面评论: {cover_comment}")
    print(f"接收到的视频文案: {script_content}")

    # Step 2: Get trending scores using get_bewrite_util
    trending, emotion, visual, creativity = get_bewrite_util(video_title, cover_comment, config)
    logger.info(
        f"趋势分析结果: trending={trending:.2f}, emotion={emotion:.2f}, visual={visual:.2f}, creativity={creativity:.2f}"
    )
    print(
        f"趋势分析结果: trending={trending:.2f}, emotion={emotion:.2f}, visual={visual:.2f}, creativity={creativity:.2f}"
    )

    # Step 3: Prepare and return response
    response_data = {
        "step": "封面分析",
        "message": "✅ 封面分析成功",
        "trending_analysis": {
            "trending": f"{trending:.2f}",
            "emotion": f"{emotion:.2f}",
            "visual": f"{visual:.2f}",
            "creativity": f"{creativity:.2f}"
        },
        "script_content": script_content  # Include script content in response for frontend use
    }

    logger.info(f"返回前端数据: {response_data}")
    print(f"返回前端数据: {response_data}")

    return jsonify(response_data)


def handle_model_evaluation():
    """
    Handle /analyze_model_evaluation POST request:
    1. Extract settings and other parameters from request
    2. Call call_predict_api with the provided data
    3. Return the prediction result to the frontend
    """
    # Step 1: Extract settings and parameters
    req_data = request.get_json()
    if not req_data:
        return jsonify({"error": "缺少请求数据"}), 400

    settings = req_data.get("settings", {})
    if not settings:
        return jsonify({"error": "缺少参数: settings"}), 400

    aiserverurl = settings.get("aiServerUrl", "http://192.168.1.3:10650/")
    video_title = req_data.get("视频标题", "")
    cover_comment = req_data.get("封面评论", "")
    category = req_data.get("分区", "")
    follower_count = req_data.get("粉丝数", "")
    p_rating = req_data.get("p_rating", 0)
    duration = req_data.get("时长（秒）", 0)
    trending = req_data.get("trending", "0.00")
    emotion = req_data.get("emotion", "0.00")
    visual = req_data.get("visual", "0.00")
    creativity = req_data.get("creativity", "0.00")

    # Validate required fields
    required_fields = {
        "视频标题": video_title,
        "封面评论": cover_comment,
        "分区": category,
        "粉丝数": follower_count,
    }
    missing_fields = [field for field, value in required_fields.items() if not value]
    if missing_fields:
        logger.error(f"缺少必要的字段: {', '.join(missing_fields)}")
        return jsonify({"error": f"缺少必要的字段: {', '.join(missing_fields)}"}), 400

    # Type conversion and validation
    try:
        follower_count = int(follower_count)
        p_rating = float(p_rating)
        duration = int(duration)
        trending = float(trending)
        emotion = float(emotion)
        visual = float(visual)
        creativity = float(creativity)
    except (ValueError, TypeError) as e:
        logger.error(f"参数类型转换失败: {e}")
        return jsonify({"error": f"参数类型无效: {str(e)}"}), 400

    logger.info(f"接收到的参数: video_title={video_title}, cover_comment={cover_comment}, category={category}, "
                f"follower_count={follower_count}, p_rating={p_rating}, duration={duration}, "
                f"trending={trending}, emotion={emotion}, visual={visual}, creativity={creativity}")
    print(f"接收到的参数: video_title={video_title}, cover_comment={cover_comment}, category={category}, "
          f"follower_count={follower_count}, p_rating={p_rating}, duration={duration}, "
          f"trending={trending}, emotion={emotion}, visual={visual}, creativity={creativity}")

    # Step 2: Call predict API
    predict_result = call_predict_api(
        aiserverurl,
        video_title,
        cover_comment,
        category,
        follower_count,
        p_rating,
        duration,
        trending,
        emotion,
        visual,
        creativity
    )
    if not predict_result:
        logger.error("调用预测接口失败")
        return jsonify({"error": "调用预测接口失败"}), 500

    # Step 3: Prepare and return response
    response_data = {
        "step": "模型评估",
        "message": "✅ 模型评估成功",
        "predicted_play_count": predict_result.get("predicted_play_count"),
        "raw_regression": predict_result.get("raw_regression"),
        "post_processed": predict_result.get("post_processed")
    }

    logger.info(f"返回前端数据: {response_data}")
    print(f"返回前端数据: {response_data}")

    return jsonify(response_data)


if __name__ == "__main__":
    # For testing purposes, you might need to mock Flask's request object
    pass
