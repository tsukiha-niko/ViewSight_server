import base64
import io
import json
import logging
import os
import random
import subprocess
import tempfile

import torch
import whisperx
from flask import request, jsonify
from werkzeug.datastructures import FileStorage

from image_handler import handle_img  # 导入新函数
from utils.get_comment_util import get_video_summary_util


def format_time(seconds):
    """将秒数转换为 hh:mm:ss 格式"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def get_video_duration(file_path):
    """使用ffmpeg获取视频时长（返回秒数）"""
    result = subprocess.run(['ffmpeg', '-i', file_path], capture_output=True, text=True)
    for line in result.stderr.split('\n'):
        if 'Duration' in line:
            time_str = line.split('Duration: ')[1].split(',')[0]
            h, m, s = map(float, time_str.split(':'))
            return h * 3600 + m * 60 + s
    logging.error("无法获取视频时长")
    return 0


def extract_frame(file_path, timestamp, output_path):
    """使用ffmpeg提取特定时间的帧"""
    subprocess.run([
        'ffmpeg', '-i', file_path, '-ss', str(timestamp),
        '-frames:v', '1', '-q:v', '2', output_path, '-y'
    ], capture_output=True)


def slice_video(file_path, srt_content):
    """根据字幕切片视频并返回图片数据"""
    duration = get_video_duration(file_path)
    if not duration:
        return []

    # 解析SRT内容
    segments = []
    if srt_content:
        lines = srt_content.split('\n\n')
        for line in lines:
            parts = line.split('\n')
            if len(parts) >= 2:
                time_range = parts[1].strip('[]').split(' - ')
                start = sum(x * float(t) for x, t in zip([3600, 60, 1], time_range[0].split(':')))
                end = sum(x * float(t) for x, t in zip([3600, 60, 1], time_range[1].split(':')))
                segments.append({'start': start, 'end': end, 'text': parts[2] if len(parts) > 2 else ''})

    # 决定切片数量
    num_segments = len(segments)
    if num_segments > 12:
        segments = random.sample(segments, 12)
    elif num_segments < 5 or not segments:
        segments = [{'start': i * duration / 5, 'end': (i + 1) * duration / 5, 'text': f'Segment {i + 1}'}
                    for i in range(5)]

    # 提取帧
    frames = []
    temp_dir = tempfile.gettempdir()
    for i, seg in enumerate(segments):
        timestamp = (seg['start'] + seg['end']) / 2
        output_path = os.path.join(temp_dir, f'frame_{i}.jpg')
        extract_frame(file_path, timestamp, output_path)

        with open(output_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        frames.append({
            'data_url': f'data:image/jpeg;base64,{img_data}',
            'timestamp': format_time(timestamp),
            'text': seg['text']
        })
        os.remove(output_path)

    return frames


def analyze_frames(frames, duration):
    """使用AI分析帧并总结视频内容"""
    analysis_results = []
    original_files = request.files  # 保存原始request.files

    for i, frame in enumerate(frames):
        prompt = (
            f"这是一个视频节选的第{i + 1}帧（共{len(frames)}帧），时间戳：{frame['timestamp']}，"
            f"对应字幕：'{frame['text']}'，视频总时长：{int(duration)}秒。请详细分析以下内容："
            f"1. 构图：主体位置、背景元素、画面布局；"
            f"2. 色彩风格：明暗对比、色调倾向、视觉氛围；"
            f"3. 显著特征：人物、物体、文字等关键元素；"
            f"4. 情感与意图：画面传达的情绪或叙事目的；"
            f"5. 建议：结合时间戳和字幕，提出优化建议（如文案调整、时长分配、画面编排）。"
            f"请尽量细致描述画面内容，并推测其在视频中的作用与上下文关联。"
        )

        # 将base64转换为FileStorage对象
        img_bytes = base64.b64decode(frame['data_url'].split(',')[1])
        file_storage = FileStorage(
            stream=io.BytesIO(img_bytes),
            filename=f'frame_{i}.jpg',
            content_type='image/jpeg'
        )
        request.files = {'file': file_storage}

        result = handle_img(prompt)
        if isinstance(result, tuple) and result[1] == 200:
            analysis = result[0].get_json()['img_info']
            analysis_results.append({
                'frame': i + 1,
                'timestamp': frame['timestamp'],
                'subtitle': frame['text'],
                'analysis': analysis
            })

    request.files = original_files  # 恢复原始request.files

    text = "段落关键" if len(frames) <= 5 else "随机"

    # 获取config
    config = {}
    if request.form.get('config'):
        try:
            config = json.loads(request.form['config'])  # 使用json.loads解析config
        except Exception as e:
            logging.error(f"解析config失败: {str(e)}")

    # 调用get_video_summary_util生成总结
    summary_prompt = (
        f"这是对视频{len(frames)}个{text}的分析结果，视频总时长为{int(duration)}秒，字幕内容如下：\n"
        f"{''.join([f'[{f['timestamp']}] {f['text']}\n' for f in frames])}"
        f"请根据帧分析和字幕，推断视频的主题、内容概要、场景切换和叙事风格，并提供以下建议："
        f"1. 时长优化：是否需要调整总时长或某段时长；"
        f"2. 分区建议：如何划分视频段落以提升节奏；"
        f"3. 编排改进：画面顺序或过渡的优化建议；"
        f"4. 文案调整：字幕或旁白的改进方向。"
        f"总结控制在200字以内，建议简洁实用。无需返回markdown语法,分段讲述即可"
    )
    summary = get_video_summary_util(config, analysis_results, summary_prompt)

    return analysis_results, summary


def transcribe_video(file_path):
    """使用WhisperX转录视频音频"""
    logging.info("开始使用 whisper X 进行视频转录...")

    def get_optimal_device():
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    device = get_optimal_device()
    print(f"自动选择设备: {device}")
    compute_type = "float16" if device == "cuda" else "float32"

    try:
        model = whisperx.load_model("medium", device=device, compute_type=compute_type)
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        model = whisperx.load_model("small", device="cpu", compute_type="float32")

    try:
        result = model.transcribe(file_path)
        segments = result.get("segments", [])
        if not segments:
            logging.warning("未检测到有效语音，将返回空转录")
            return ""
    except Exception as e:
        logging.error(f"转录失败: {str(e)}")
        return ""

    merged_segments = []
    current_seg = segments[0] if segments else {'start': 0, 'end': 0, 'text': ''}
    for seg in segments[1:]:
        if seg["start"] - current_seg["end"] <= 5:
            current_seg["end"] = seg["end"]
            current_seg["text"] += " " + seg["text"]
        else:
            merged_segments.append(current_seg)
            current_seg = seg
    if segments:
        merged_segments.append(current_seg)

    srt_result = ""
    for idx, seg in enumerate(merged_segments, start=1):
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        srt_result += f"{idx}\n[{start_time} - {end_time}]\n{seg['text']}\n\n"
    logging.info("视频转录完成")
    return srt_result.strip()


def handle_video():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "未提供视频文件"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "视频文件名为空"}), 400

        # 检查是否有scriptContent
        script_content = None
        if request.form.get('config'):
            try:
                script_content = request.form.get('scriptContent', '').strip()
            except Exception as e:
                logging.error(f"解析config失败: {str(e)}")

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        logging.info("视频文件已临时保存：%s", temp_path)

        # 获取视频时长
        duration = get_video_duration(temp_path)

        # 如果有scriptContent，则直接使用，否则进行转录
        if script_content:
            logging.info("使用提供的scriptContent作为srt_info")
            transcription = script_content
        else:
            logging.info("未提供scriptContent，将进行WhisperX转录")
            transcription = transcribe_video(temp_path)

        frames = slice_video(temp_path, transcription)
        analysis_results, summary = analyze_frames(frames, duration)

        final_message = "✅ 视频分析成功"
        logging.info("视频处理成功")

        os.remove(temp_path)
        logging.info("已删除临时视频文件")

        return jsonify({
            "step": "视频上传",
            "message": final_message,
            "srt_info": transcription,
            "duration_seconds": int(duration),
            "frame_analysis": analysis_results,
            "video_summary": summary
        }), 200
    except Exception as e:
        logging.exception("视频处理出错")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For testing purposes
    pass
