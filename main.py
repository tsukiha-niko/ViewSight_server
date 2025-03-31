import logging

import torch
from flask import Flask
from flask_cors import CORS

import config_handler
import image_handler
import title_handler
import video_handler
from blink_handler import handle_blink
from bvideo_handler import handle_bvideo, handle_model_evaluation  # Import both handlers

app = Flask(__name__)
CORS(app)  # Allow all cross-origin requests
logging.basicConfig(level=logging.INFO)

print(torch.cuda.is_available)


@app.route('/up_config', methods=['POST'])
def up_config():
    return config_handler.handle_config()


@app.route('/up_title', methods=['POST'])
def up_title():
    return title_handler.handle_title()


@app.route('/up_img', methods=['POST'])
def up_img():
    return image_handler.handle_img()


@app.route('/up_video', methods=['POST'])
def up_video():
    return video_handler.handle_video()


@app.route('/analyze_video_link', methods=['POST'])
def get_blink_analyse():
    return handle_blink()


@app.route('/analyze_video_cover', methods=['POST'])
def analyze_video_cover():
    return handle_bvideo()


@app.route('/analyze_model_evaluation', methods=['POST'])
def analyze_model_evaluation():
    return handle_model_evaluation()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4432)
