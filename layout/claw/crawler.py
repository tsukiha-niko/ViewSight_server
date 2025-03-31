import configparser
import csv
import hashlib
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"
]

headers = {
    'User-Agent': random.choice(USER_AGENTS),
    'Referer': 'https://www.bilibili.com/'
}


def slow_request(url, headers, params=None):
    time.sleep(random.uniform(2, 3.0))  # 增加延迟以降低风险
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"请求失败: {str(e)}")
        return None


def filter_video(tname, articles_average_likes):
    if any(keyword in tname for keyword in ["游戏", "搞笑", "舞蹈"]):
        return random.random() < 0.15
    elif "日常" in tname:
        return random.random() < 0.25
    if articles_average_likes < 0.1:
        return random.random() < 0.01
    elif articles_average_likes < 1:
        return random.random() < 0.1
    return True


def load_cookie(config_filename="config.properties"):
    config = configparser.ConfigParser()
    try:
        config.read(config_filename, encoding='utf-8')
        cookie = config.get('bilibili', 'cookie', raw=True, fallback=None)
        if cookie:
            logger.info("Cookie 加载成功")
            return cookie
        else:
            logger.warning("配置文件中未找到 cookie")
            return None
    except Exception as e:
        logger.error(f"读取配置文件失败: {e}")
        return None


def get_wbi_params(cookie=None):
    url = 'https://api.bilibili.com/x/web-interface/nav'
    if cookie:
        headers['Cookie'] = cookie
    try:
        response = slow_request(url, headers)
        if not response:
            return None
        data = response.json()
        if data['code'] != 0:
            logger.error(f"获取 WBI 参数失败: code={data['code']}, message={data.get('message', '未知错误')}")
            return None
        wbi_img = data['data']['wbi_img']
        img_key = wbi_img['img_url'].split('/')[-1].split('.')[0]
        sub_key = wbi_img['sub_url'].split('/')[-1].split('.')[0]
        logger.info(f"成功获取 WBI 参数: img_key={img_key}, sub_key={sub_key}")
        return {'img_key': img_key, 'sub_key': sub_key, 'wts': int(time.time())}
    except Exception as e:
        logger.error(f"获取 WBI 参数出错: {str(e)}")
        return None


def generate_w_rid(params, img_key, sub_key):
    mixin_key_enc_tab = [
        46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
        33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
        61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
        36, 20, 34, 44, 52
    ]
    combined_key = img_key + sub_key
    mixin_key = ''.join(combined_key[i] for i in mixin_key_enc_tab)[:32]
    params_list = sorted([f"{k}={v}" for k, v in params.items()])
    params_str = '&'.join(params_list)
    w_rid = hashlib.md5((params_str + mixin_key).encode('utf-8')).hexdigest()
    return w_rid


def get_latest_videos(cookie=None, start_page=100, pages_to_fetch=10, csv_filename='bilibili_videos.csv'):
    video_list = []
    base_url = 'https://api.bilibili.com/x/web-interface/newlist'
    if cookie:
        headers['Cookie'] = cookie

    existing_bvids = set()
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_bvids = {row['bvid'] for row in reader}

    for page in range(start_page, start_page + pages_to_fetch):
        params = {'pn': page, 'ps': 20}
        response = slow_request(base_url, headers, params)
        if not response:
            continue
        data = response.json()
        if data['code'] == 0 and 'data' in data and 'archives' in data['data']:
            new_videos = []
            for video in data['data']['archives']:
                bvid = video['bvid']
                if bvid not in existing_bvids:
                    new_videos.append([0, '', '', bvid, '', 0, 0, 0.0, '', f"https://www.bilibili.com/video/{bvid}"])
                    video_list.append(bvid)
            if new_videos:
                save_to_csv(new_videos, csv_filename, is_first_write=not os.path.exists(csv_filename))
                logger.info(f"已获取并保存第 {page} 页视频列表，新增 {len(new_videos)} 个视频")
        else:
            logger.error(f"第 {page} 页返回异常: {data.get('message', '未知错误')}")
    return video_list[:10000]


def get_video_detail(bvid, cookie, wbi_params):
    if not wbi_params or 'img_key' not in wbi_params or 'sub_key' not in wbi_params:
        logger.warning(f"BV{bvid} 跳过 - WBI 参数不完整: {wbi_params}")
        return None
    url = 'https://api.bilibili.com/x/web-interface/wbi/view/detail'
    params = {'bvid': bvid, 'wts': wbi_params['wts']}
    if cookie:
        headers['Cookie'] = cookie
    w_rid = generate_w_rid(params, wbi_params['img_key'], wbi_params['sub_key'])
    params['w_rid'] = w_rid
    response = slow_request(url, headers, params)
    if not response:
        return None
    data = response.json()
    if data['code'] != 0:
        logger.error(f"BV{bvid} API返回错误: code={data['code']}, message={data.get('message', '未知错误')}")
        return None
    if 'data' not in data:
        logger.error(f"BV{bvid} 返回数据缺少 'data' 字段: {json.dumps(data)}")
        return None
    return data['data']


def save_to_csv(data_list, filename='bilibili_videos.csv', is_first_write=False):
    mode = 'w' if is_first_write else 'a'
    with open(filename, mode, newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        if is_first_write:
            writer.writerow(['fan_count', 'title', 'cover_url', 'bvid', 'tags', 'archive_count', 'like_num',
                             'articles_average_likes', 'tname', 'link'])
        for row in data_list:
            writer.writerow(row)


def process_video(bvid, cookie, wbi_params):
    logger.info(f"正在处理视频: {bvid}")
    video_data = get_video_detail(bvid, cookie, wbi_params)
    if video_data:
        try:
            fan_count = int(video_data['Card'].get('follower', 0))
            archive_count = int(video_data['Card'].get('archive_count', 0))
            like_num = int(video_data['Card'].get('like_num', 0))
            title = video_data['View']['title']
            cover_url = video_data['View']['pic']
            tname = video_data['View'].get('tname', '未知分类')
            tags = '、'.join([tag['tag_name'] for tag in video_data.get('Tags', [])])
            articles_average_likes = round(like_num / archive_count, 2) if archive_count > 0 else 0.0
            link = f"https://www.bilibili.com/video/{bvid}"
            if not filter_video(tname, articles_average_likes):
                logger.info(f"BV{bvid} 被筛选掉: tname={tname}, articles_average_likes={articles_average_likes}")
                return None
            return [fan_count, title, cover_url, bvid, tags, archive_count, like_num, articles_average_likes, tname,
                    link]
        except Exception as e:
            logger.error(f"BV{bvid} 数据处理失败: {str(e)}")
    return None


def update_existing_csv(cookie, wbi_params, csv_filename='bilibili_videos.csv'):
    if not os.path.exists(csv_filename):
        logger.warning("CSV 文件不存在，无法更新")
        return
    bvids_to_update = []
    with open(csv_filename, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row['title']:  # 如果 title 为空，认为需要更新
                bvids_to_update.append(row['bvid'])

    updated_data = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_video, bvid, cookie, wbi_params) for bvid in bvids_to_update]
        for future in futures:
            result = future.result()
            if result:
                updated_data.append(result)
                if len(updated_data) >= 10:
                    save_to_csv(updated_data, csv_filename, is_first_write=False)
                    updated_data = []
                    logger.info(f"已更新 {len(updated_data)} 条数据到 CSV")

    if updated_data:
        save_to_csv(updated_data, csv_filename, is_first_write=False)
        logger.info(f"已更新剩余 {len(updated_data)} 条数据到 CSV")


def main():
    cookie = load_cookie()
    if not cookie:
        logger.warning("未加载到 Cookie，可能影响数据获取")
    wbi_params = get_wbi_params(cookie)
    if not wbi_params:
        logger.error("WBI 参数获取失败，程序退出")
        return

    csv_filename = 'bilibili_videos.csv'
    if os.path.exists(csv_filename):
        logger.info("检测到已有 CSV 文件，开始更新现有数据...")
        update_existing_csv(cookie, wbi_params, csv_filename)
    else:
        logger.info("开始获取最新视频列表并创建 CSV...")
        bv_list = get_latest_videos(cookie, start_page=10000, pages_to_fetch=100, csv_filename=csv_filename)
        update_existing_csv(cookie, wbi_params, csv_filename)

    logger.info("处理完成！数据已保存到 bilibili_videos.csv")


if __name__ == "__main__":
    main()
