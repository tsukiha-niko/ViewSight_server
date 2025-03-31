import requests


def call_predict_api(aiserverurl, video_title, cover_comment, category, follower_count, p_rating, duration, trending,
                     emotion, visual, creativity):
    """Call the prediction API with video details including trending scores"""
    predict_url = f"{aiserverurl.rstrip('/')}/predict"
    params = {
        "视频标题": video_title,
        "封面评论": cover_comment,
        "分区": category,
        "粉丝数": follower_count,
        "p_rating": p_rating,
        "时长（秒）": duration,
        "trending": trending,  # Add trending score
        "emotion": emotion,  # Add emotion score
        "visual": visual,  # Add visual score
        "creativity": creativity  # Add creativity score
    }
    try:
        resp = requests.get(predict_url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return e
