import requests


def get_bewrite_util(video_title: str, cover_comment: str, config: dict) -> tuple[float, float, float, float]:
    """
    Analyze the trending relevance of a single video title and cover comment.

    Args:
        video_title (str): The title of the video
        cover_comment (str): The comment on the video cover
        config (dict): Configuration JSON containing API endpoints and credentials

    Returns:
        tuple: (trending, emotion, visual, creativity) scores between 0 and 1
    """
    # Extract required fields from config
    text_url = config.get("textUrl", "")
    text_token = config.get("textToken", "")
    text_model = config.get("textModel", "")

    # Fallback to image-related config if text config is not provided
    if not text_url or not text_token or not text_model:
        text_url = config.get("imageUrl", "https://api.siliconflow.cn/v1/chat/completions")
        text_token = config.get("imageToken", "")
        text_model = config.get("imageModel", "Qwen/Qwen2.5-VL-32B-Instruct")

    # Construct the prompt for a single title-comment pair
    prompt = (
        "请分析以下视频标题和封面评论的趋势相关性，严格按照以下要求返回结果：\n"
        "1. **仅返回纯字符串，不包含任何 JSON 标记、多余文本、注释或说明**。\n"
        "2. 返回1组评分，包含4个字段，用空格分隔，顺序为：trending emotion visual creativity。\n"
        "3. 每个字段值为0到1之间的小数（保留2位小数）。\n"
        "   - trending: 时势热点（标题和封面是否紧跟当前热点，吸引关注）。\n"
        "   - emotion: 情感共鸣（标题和封面能否引起情感反应，如好奇、兴奋、感动）。\n"
        "   - visual: 视觉吸引力（封面设计是否吸引，能否脱颖而出）。\n"
        "   - creativity: 创意与独特性（标题和封面是否独特，提供新鲜感）。\n"
        "4. 确保输出格式正确，所有数值在 0-1 范围内，无非法字符。\n"
        f"输入数据：{{'title': '{video_title}', 'comment': '{cover_comment}'}}\n"
        "示例输出：\n"
        "0.86 0.77 0.97 0.83"
    )

    # Prepare API payload
    payload = {
        "model": text_model,
        "stream": False,
        "max_tokens": 4096,
        "temperature": 0.3,
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
        "Authorization": f"Bearer {text_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(text_url, headers=headers, json=payload, timeout=90)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            values = content.split()
            if len(values) != 4:
                print(f"❌ 输出格式错误: {content}, 期望4个字段")
                return (0.00, 0.00, 0.00, 0.00)

            scores = [float(v) for v in values]
            if not all(0 <= s <= 1 for s in scores):
                print(f"❌ 评分超出范围: {content}")
                return (0.00, 0.00, 0.00, 0.00)

            return (scores[0], scores[1], scores[2], scores[3])
        else:
            print(f"❌ API 返回错误 [{response.status_code}]：{response.text}")
            return (0.00, 0.00, 0.00, 0.00)
    except Exception as e:
        print(f"❌ API 分析异常：{e}")
        return (0.00, 0.00, 0.00, 0.00)
