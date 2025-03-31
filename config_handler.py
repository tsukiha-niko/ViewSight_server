import json
import logging
import os

from flask import request, jsonify

# 配置文件存储在程序根目录下
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')


def handle_config():
    try:
        config_data = request.get_json()
        if not config_data or not config_data.get('backendUrl', '').strip():
            return jsonify({"error": "后端地址是必填项！"}), 400

        # 将配置写入到根目录下的 config.json 文件
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        logging.info("配置已保存：%s", config_data)
        return jsonify({"message": "配置已成功保存到本地及应用到服务器"}), 200
    except Exception as e:
        logging.exception("保存配置时出错")
        return jsonify({"error": str(e)}), 500


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}
