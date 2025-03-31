**1. 配置应用程序（已废弃）**

*   **端点:** `/up_config`
*   **方法:** `POST`
*   **目的:** 将应用程序的配置设置保存或更新到本地的 `config.json` 文件。
*   **请求:**
    *   **Content-Type:** `application/json`
    *   **Body:** 包含配置键值对的 JSON 对象。
        *   `backendUrl` (string, 必需): 后端服务器地址。
        *   `imageUrl` (string, 可选): 图像分析 API 的 URL。
        *   `imageToken` (string, 可选): 图像分析 API 的 Token。
        *   `imageModel` (string, 可选): 用于图像分析的模型名称。
        *   `cookie` (string, 可选): 用于访问 Bilibili API 的 Cookie。
        *   `aiServerUrl` (string, 可选): 预测模型 API 的 URL。
        *   *(其他需要的配置项)*
*   **成功响应 (200 OK):**
    *   **Content-Type:** `application/json`
    *   **Body:**
        ```json
        {
          "message": "配置已成功保存到本地及应用到服务器"
        }
        ```
*   **错误响应:**
    *   **400 Bad Request:** 缺少或无效的 `backendUrl`。
        ```json
        { "error": "后端地址是必填项！" }
        ```
    *   **500 Internal Server Error:** 文件写入时出错。
        ```json
        { "error": "<具体的错误信息>" }
        ```

---

**2. 分析标题 (基础)**

*   **端点:** `/up_title`
*   **方法:** `POST`
*   **目的:** 接收视频标题（目前仅做简单处理，主要是记录日志与测试联络性）。
*   **请求:**
    *   **Content-Type:** `application/json`
    *   **Body:**
        ```json
        {
          "title": "你的视频标题" // (string, 必需)
        }
        ```
*   **成功响应 (200 OK):**
    *   **Content-Type:** `application/json`
    *   **Body:**
        ```json
        {
          "step": "标题上传",
          "message": "✅ 标题分析成功"
        }
        ```
*   **错误响应:**
    *   **400 Bad Request:** 缺少标题。
        ```json
        { "error": "标题不能为空" }
        ```
    *   **500 Internal Server Error:** 通用处理错误。
        ```json
        { "error": "<具体的错误信息>" }
        ```

---

**3. 分析图像 (封面)**

*   **端点:** `/up_img`
*   **方法:** `POST`
*   **目的:** 使用外部 AI 服务分析图像（视频封面）。接受通过文件上传、URL 或 Base64 编码的图像。根据图像的 MD5 哈希缓存结果。
*   **请求:**
    *   **Content-Type:** `multipart/form-data`
    *   **Form Data:**
        *   `file`: 图像文件本身 (文件上传)，或包含图像 URL 的字符串，或包含 Base64 编码图像数据的字符串 (必需, 必须提供其中一种格式)。
        *   `config`: 包含配置的 JSON 字符串 (必需)。必须包含 `imageUrl`, `imageToken`, `imageModel`。
        *   `prompt` (string, 可选): 用于图像分析 AI 的自定义提示。如果省略，则使用默认提示。因大数据模型特性，推荐使用默认提示，理论上留空效果最佳。
*   **成功响应 (200 OK):**
    *   **Content-Type:** `application/json`
    *   **Body:**
        ```json
        {
          "step": "封面上传",
          "message": "✅ 封面分析成功", // 或 "✅ 封面分析成功 (使用缓存数据)"
          "img_info": "AI 分析结果文本...", // (string) 来自 AI 模型的分析文本。
          "cached": true/false, // (boolean) 结果是否来自缓存。
          "md5_hash": "..." // (string) 已分析图像的 MD5 哈希值。
        }
        ```
*   **错误响应:**
    *   **400 Bad Request:** 缺少文件/数据、无效的 Base64、不支持的图像类型、缺少/不完整的配置、图像下载失败。
        ```json
        { "error": "<具体的错误信息>" } // 例如: "未提供封面文件", "配置未提供", "Base64数据格式错误"
        ```
    *   **500 Internal Server Error:** AI API 调用失败、读/写缓存错误、其他处理错误。
        ```json
        { "error": "<具体的错误信息>" } // 例如: "图片处理失败: <状态码>"
        ```

---

**4. 分析视频 (文件上传)**

*   **端点:** `/up_video`
*   **方法:** `POST`
*   **目的:** 分析上传的视频文件。执行转录（如果未提供 `scriptContent`，则使用 WhisperX），提取帧，使用图像分析服务（`/up_img` 逻辑）分析帧，并生成摘要。
*   **请求:**
    *   **Content-Type:** `multipart/form-data`
    *   **Form Data:**
        *   `file`: 视频文件 (必需)。
        *   `config`: 包含配置的 JSON 字符串 (必需，用于通过 `handle_img` 进行帧分析)。必须包含 `imageUrl`, `imageToken`, `imageModel`。也用于 `get_video_summary_util`。
        *   `scriptContent` (string, 可选): 已有的脚本/字幕内容。如果提供，则跳过 WhisperX 转录，提高处理速度。
*   **成功响应 (200 OK):**
    *   **Content-Type:** `application/json`
    *   **Body:**
        ```json
        {
          "step": "视频上传",
          "message": "✅ 视频分析成功",
          "srt_info": "1\n[00:00:00 - 00:00:05]\n字幕文本...\n\n2\n...", // (string) 生成的或提供的字幕内容。
          "duration_seconds": 123, // (integer) 视频时长（秒）。
          "frame_analysis": [ // (array) 关键帧的分析结果。
            {
              "frame": 1,
              "timestamp": "00:00:10",
              "subtitle": "该帧附近的文本...",
              "analysis": "第1帧的 AI 分析..."
            },
            // ... 更多帧
          ],
          "video_summary": "整体视频摘要和建议..." // (string) 基于帧分析的 AI 生成摘要。
        }
        ```
*   **错误响应:**
    *   **400 Bad Request:** 缺少视频文件、文件名为空。
        ```json
        { "error": "<具体的错误信息>" } // 例如: "未提供视频文件"
        ```
    *   **500 Internal Server Error:** 保存/删除临时文件错误、FFmpeg 错误、WhisperX 错误、帧分析错误、摘要生成错误。
        ```json
        { "error": "<具体的错误信息>" }
        ```

---

**5. 分析 Bilibili 链接 (基础)**

*   **端点:** `/analyze_video_link`
*   **方法:** `POST`
*   **目的:** 分析 Bilibili 视频链接（或 BV/AV 号）。获取视频元数据，分析封面，获取趋势评分，计算 UP 主评分，使用模型预测播放量，并计算播放量范围。
*   **请求:**
    *   **Content-Type:** `application/json`
    *   **Body:**
        ```json
        {
          "videoLink": "https://www.bilibili.com/video/BVxxxxxxx/", // (string, 必需) Bilibili 视频 URL 或 BV/AV 号。
          "config": { // (object, 必需) 应用配置。
            "cookie": "你的 Bilibili Cookie", // (string, 必需，用于获取视频/UP主信息)
            "aiServerUrl": "http://192.168.1.3:10650", // (string, 必需，用于预测)
            "imageUrl": "...", // (string, 必需，用于封面分析)
            "imageToken": "...", // (string, 必需，用于封面分析)
            "imageModel": "...", // (string, 必需，用于封面分析)
            "backendUrl": "...", // (string, 必需，用于趋势分析)
            "token": "...", // (string, 必需，用于趋势分析)
            "model": "..." // (string, 必需，用于趋势分析)
            // ... 如果 utils 需要其他配置
          }
        }
        ```
*   **成功响应 (200 OK):**
    *   **Content-Type:** `application/json`
    *   **Body:**
        ```json
        {
          "step": "链接解析",
          "message": "✅ 基础解析成功\n视频分析请移步专业版",
          "predicted_play_count": 15000.0, // (float) 模型原始预测播放量。
          "range_score": "1500 - 150000", // (string) 估算的播放量范围（对数刻度）。
          "estimated7DayViews": "4743 - 47434", // (string) 估算的播放量范围 (+/- 0.5 log10)。
          "raw_regression": 4.176, // (float) 回归模型的原始输出。
          "post_processed": 15000.0, // (float) 后处理的预测值。
          "trending_analysis": { // (object) 来自标题/封面分析的评分。
             "trending": "0.85",
             "emotion": "0.70",
             "visual": "0.90",
             "creativity": "0.65"
           }
        }
        ```
*   **错误响应:**
    *   **400 Bad Request:** 缺少 `videoLink` 或 `config`，无效的链接/ID。
        ```json
        { "error": "<具体的错误信息>" }
        ```
    *   **500 Internal Server Error:** 从 Bilibili API 获取信息失败、封面/趋势分析失败、预测 API 调用失败。
        ```json
        { "error": "<具体的错误信息>" } // 例如: "获取视频信息失败", "调用预测接口失败"
        ```

---

**6. 分析视频封面/标题 (趋势评分)**

*   **端点:** `/analyze_video_cover`
*   **方法:** `POST`
*   **目的:** 分析提供的视频标题和封面评论（推测是从上一步或外部来源获取），以使用 `get_bewrite_util` 生成趋势、情感、视觉和创意评分。
*   **请求:**
    *   **Content-Type:** `application/json`
    *   **Body:**
        ```json
        {
          "combined_info": { // (object, 必需)
            "视频标题": "你的视频标题", // (string, 必需)
            "封面评论": "封面的 AI 分析结果", // (string, 必需) 封面的文本描述/分析。
            "视频文案": "可选的脚本内容..." // (string, 可选) 视频脚本内容。
          },
          "config": { // (object, 必需) 用于 get_bewrite_util 的配置。
             "backendUrl": "...", // (string, 必需)
             "token": "...", // (string, 必需)
             "model": "..." // (string, 必需)
             // ... 如果 get_bewrite_util 需要其他配置
          }
        }
        ```
*   **成功响应 (200 OK):**
    *   **Content-Type:** `application/json`
    *   **Body:**
        ```json
        {
          "step": "封面分析",
          "message": "✅ 封面分析成功",
          "trending_analysis": {
            "trending": "0.85", // (string) 格式化后的分数。
            "emotion": "0.70", // (string) 格式化后的分数。
            "visual": "0.90", // (string) 格式化后的分数。
            "creativity": "0.65" // (string) 格式化后的分数。
          },
          "script_content": "可选的脚本内容..." // (string) 从输入透传。
        }
        ```
*   **错误响应:**
    *   **400 Bad Request:** 缺少 `combined_info` 或 `config`，`combined_info` 中缺少必需字段。
        ```json
        { "error": "<具体的错误信息>" }
        ```
    *   **500 Internal Server Error:** 在 `get_bewrite_util` 中分析时出错。
        ```json
        { "error": "<具体的错误信息>" }
        ```

---

**7. 评估模型预测**

*   **端点:** `/analyze_model_evaluation`
*   **方法:** `POST`
*   **目的:** 接收详细的视频元数据，包括预先计算的趋势评分（可能来自 `/analyze_video_cover`），并调用预测 API (`call_predict_api`) 以获取播放量预测。
*   **请求:**
    *   **Content-Type:** `application/json`
    *   **Body:**
        ```json
        {
          "settings": { // (object, 必需) 包含 AI 服务器 URL。
            "aiServerUrl": "http://192.168.1.3:10650/" // (string, 必需)
          },
          "视频标题": "你的视频标题", // (string, 必需)
          "封面评论": "封面的 AI 分析结果", // (string, 必需) 文本描述。
          "分区": "知识", // (string, 必需) 视频分区。
          "粉丝数": "10000", // (string/integer, 必需) 粉丝数。
          "p_rating": 0.5, // (float/integer, 可选, 默认为 0) UP主的平均点赞/稿件比。
          "时长（秒）": 300, // (integer, 可选, 默认为 0) 视频时长。
          "trending": "0.85", // (string/float, 必需) 趋势评分。
          "emotion": "0.70", // (string/float, 必需) 情感评分。
          "visual": "0.90", // (string/float, 必需) 视觉评分。
          "creativity": "0.65" // (string/float, 必需) 创意评分。
        }
        ```
*   **成功响应 (200 OK):**
    *   **Content-Type:** `application/json`
    *   **Body:**
        ```json
        {
          "step": "模型评估",
          "message": "✅ 模型评估成功",
          "predicted_play_count": 15000.0, // (float) 模型预测的播放量。
          "raw_regression": 4.176, // (float) 回归模型的原始输出。
          "post_processed": 15000.0 // (float) 后处理的预测值。
        }
        ```
*   **错误响应:**
    *   **400 Bad Request:** 缺少必需字段、无效的数据类型。
        ```json
        { "error": "<具体的错误信息>" }
        ```
    *   **500 Internal Server Error:** 预测 API 调用失败。
        ```json
        { "error": "调用预测接口失败" }
        ```

---

这份文档应该能清晰地说明如何与你的 Flask 应用的 API 进行交互。请记得根据你的工具函数（utils）的实际需求调整 `config` 对象的具体细节。
