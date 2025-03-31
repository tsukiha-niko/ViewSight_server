import logging

from flask import request, jsonify


def handle_title():
    try:
        data = request.get_json()
        title = data.get('title', '')
        if not title:
            return jsonify({"error": "标题不能为空"}), 400

        logging.info("收到标题: %s", title)
        # 模拟标题处理
        response_data = {
            "step": "标题上传",
            "message": "✅ 标题分析成功"
        }
        return jsonify(response_data), 200
    except Exception as e:
        logging.exception("标题处理出错")
        return jsonify({"error": str(e)}), 500
