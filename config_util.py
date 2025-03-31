import json

from flask import request


def get_config_from_request():
    config_str = None
    if request.is_json:
        data = request.get_json()
        config_str = data.get("config")
    if not config_str:
        config_str = request.form.get("config")
    if not config_str:
        return None
    try:
        if isinstance(config_str, dict):
            return config_str
        return json.loads(config_str)
    except Exception:
        return None
