import base64
import binascii
import datetime
import hashlib
import imghdr
import json
import logging
import os

import requests
from flask import request, jsonify
from werkzeug.utils import secure_filename

from config_util import get_config_from_request


def handle_img(prompt=None):
    try:
        data_url = None
        file_bytes = None
        original_filename = None

        # 1. 处理文件上传
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            original_filename = secure_filename(file.filename)
            file_bytes = file.read()
            mime = file.mimetype
            encoded = base64.b64encode(file_bytes).decode('utf-8')
            data_url = f"data:{mime};base64,{encoded}"

        # 2. 处理表单中的文件数据
        elif 'file' in request.form and request.form['file'].strip() != '':
            data_url = request.form['file'].strip()

            # 2.1 处理普通URL
            if data_url.startswith(('http://', 'https://')):
                try:
                    response = requests.get(data_url)
                    response.raise_for_status()
                    file_bytes = response.content
                    mime_type = response.headers.get('Content-Type', 'image/jpeg')
                    encoded = base64.b64encode(file_bytes).decode('utf-8')
                    data_url = f"data:{mime_type};base64,{encoded}"
                except requests.exceptions.RequestException as e:
                    logging.error(f"下载图片失败: {str(e)}")
                    return jsonify({"error": "图片下载失败"}), 400

            # 2.2 处理Base64数据
            elif not data_url.startswith('data:image/'):
                try:
                    file_bytes = base64.b64decode(data_url, validate=True)
                    image_type = imghdr.what(None, file_bytes)
                    mime_type = {
                        'jpeg': 'image/jpeg',
                        'png': 'image/png',
                        'gif': 'image/gif',
                        'bmp': 'image/bmp',
                        'webp': 'image/webp'
                    }.get(image_type)
                    if not mime_type:
                        return jsonify({"error": "无法识别的图片类型，支持JPEG/PNG/GIF/BMP/WEBP"}), 400
                    encoded = base64.b64encode(file_bytes).decode('utf-8')
                    data_url = f"data:{mime_type};base64,{encoded}"
                except binascii.Error:
                    logging.error("Base64数据无效")
                    return jsonify({"error": "Base64数据格式错误"}), 400
                except Exception as e:
                    logging.error(f"处理图片数据时出错: {e}")
                    return jsonify({"error": "图片数据处理失败"}), 400

        else:
            return jsonify({"error": "未提供封面文件"}), 400

        # 3. 计算文件MD5哈希
        if file_bytes is None:
            if data_url.startswith('data:image/'):
                file_bytes = base64.b64decode(data_url.split(',')[1])

        md5_hash = hashlib.md5(file_bytes).hexdigest()
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # 4. 检查缓存
        cache_file = os.path.join(cache_dir, 'data.json')
        cache_data = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
            except Exception as e:
                logging.error(f"读取缓存文件失败: {str(e)}")

        # 如果缓存中已经有该图片的分析结果，直接返回缓存结果
        if md5_hash in cache_data:
            try:
                logging.info(f"从缓存加载图片分析结果: {md5_hash}")
                cached_data = cache_data[md5_hash]
                return jsonify({
                    "step": "封面上传",
                    "message": "✅ 封面分析成功 (使用缓存数据)",
                    "img_info": cached_data['analysis_result'],
                    "cached": True,
                    "md5_hash": md5_hash
                }), 200
            except Exception as e:
                logging.error(f"读取缓存数据失败: {str(e)}")
                # 如果缓存数据损坏，继续进行正常处理

        # 5. 获取配置
        config = get_config_from_request()
        if not config:
            return jsonify({"error": "配置未提供"}), 400

        image_api_url = config.get('imageUrl')
        image_token = config.get('imageToken')
        image_model = config.get('imageModel')
        if not (image_api_url and image_token and image_model):
            return jsonify({"error": "图片处理配置不完整"}), 400

        # 6. 使用传入的prompt，如果没有传入，使用默认的prompt
        prompt_text = prompt or (
            "请简要分析这张视频封面，回答以下问题：图中是否有标题文字？如果有，请说出是什么文字并告诉我他的大小样式；"
            "没有则说明无文字。然后极其的简明扼要的分析画面的构图、色彩风格、吸睛元素和整体感觉。控制在100字以内。"
            "不需要使用markdown和分段说明，只回答一段段落纯文本即可"
        )

        # 7. 构造API请求
        payload = {
            "model": image_model,
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": "auto"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }],
            "stop": []
        }

        headers = {
            "Authorization": f"Bearer {image_token}",
            "Content-Type": "application/json"
        }

        # 8. 调用API
        logging.info("调用图片处理API: %s", image_api_url)
        response = requests.post(image_api_url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        if 'choices' not in result or not result['choices']:
            raise ValueError("API返回格式无效")

        content = result['choices'][0]['message'].get('content', '')

        # 9. 保存到缓存
        result_data = {
            "original_filename": original_filename,
            "md5_hash": md5_hash,
            "data_url": data_url[:100] + "..." if data_url else None,  # 只保存部分用于调试
            "analysis_result": content,
            "timestamp": datetime.datetime.now().isoformat(),
            "api_response": result  # 保存完整API响应
        }

        # 更新缓存文件
        cache_data[md5_hash] = result_data
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logging.info(f"保存图片分析结果到缓存: {cache_file}")
        except Exception as e:
            logging.error(f"保存缓存失败: {str(e)}")

        # 10. 返回结果
        return jsonify({
            "step": "封面上传",
            "message": "✅ 封面分析成功",
            "img_info": content,
            "cached": False,
            "md5_hash": md5_hash
        }), 200

    except requests.exceptions.HTTPError as e:
        logging.error("API请求失败: %s", e.response.text)
        return jsonify({"error": f"图片处理失败: {e.response.status_code}"}), 500
    except Exception as e:
        logging.exception("处理封面时出错")
        return jsonify({"error": str(e)}), 500
