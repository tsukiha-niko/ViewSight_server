import concurrent.futures
import configparser
import os
import random
import time
from datetime import datetime, timedelta

import pandas as pd
import requests


# ------------------------------
# 读取或创建配置文件 (unchanged)
# ------------------------------
def load_or_create_config(filename="config.properties"):
    config = configparser.RawConfigParser()
    default_data_dir = os.path.join(os.getcwd(), "data")
    default_recrawl_days = 7  # 默认 7 天

    if not os.path.exists(filename):
        print(f"配置文件 {filename} 不存在，创建默认配置文件...")
        config.add_section("bilibili")
        config.set("bilibili", "cookie", "")  # 默认空 cookie，用户需手动填写
        config.set("bilibili", "data_directory", default_data_dir)
        config.set("bilibili", "recrawl", str(default_recrawl_days))
        with open(filename, "w", encoding="utf-8") as configfile:
            config.write(configfile)
    else:
        config.read(filename, encoding="utf-8")

    if not config.has_section("bilibili"):
        config.add_section("bilibili")

    if not config.has_option("bilibili", "data_directory"):
        config.set("bilibili", "data_directory", default_data_dir)
        with open(filename, "w", encoding="utf-8") as configfile:
            config.write(configfile)

    if not config.has_option("bilibili", "recrawl"):
        config.set("bilibili", "recrawl", str(default_recrawl_days))
        with open(filename, "w", encoding="utf-8") as configfile:
            config.write(configfile)

    data_dir = config.get("bilibili", "data_directory")
    recrawl_days = config.getint("bilibili", "recrawl")
    os.makedirs(data_dir, exist_ok=True)
    return config, data_dir, recrawl_days


def extract_csrf_from_cookie(cookie_str):
    cookie_str = cookie_str.strip()
    if cookie_str.lower().startswith("cookie:"):
        cookie_str = cookie_str[len("cookie:"):].strip()
    for part in cookie_str.split(";"):
        if "=" in part:
            key, value = part.strip().split("=", 1)
            if key.strip() == "bili_jct":
                return value.strip()
    return ""


# ------------------------------
# 获取 UP 主数据 (unchanged)
# ------------------------------
def get_follower_counts(up_mids):
    follower_data = {}

    def fetch(mid):
        url = f"https://api.bilibili.com/x/web-interface/card?mid={mid}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Referer": "https://www.bilibili.com/"
        }
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            json_data = response.json()
            if json_data.get('code') == 0:
                data = json_data.get("data", {})
                follower_data[mid] = {
                    "粉丝数": data.get("follower", 0),
                    "历史点赞数": data.get("like_num", 0),
                    "历史稿件数": data.get("archive_count", 0)
                }
            else:
                follower_data[mid] = {"粉丝数": 0, "历史点赞数": 0, "历史稿件数": 0}
        except Exception as e:
            print(f"UP主 {mid} 数据获取失败: {e}")
            follower_data[mid] = {"粉丝数": 0, "历史点赞数": 0, "历史稿件数": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(fetch, up_mids)

    return follower_data


# ------------------------------
# 爬取“newlist”数据并写入 CSV 文件
# ------------------------------
def crawl_new_videos(rid=0, max_pages=30, view_threshold=10000, csv_filepath=""):
    if os.path.exists(csv_filepath):
        print(f"{csv_filepath} 已存在，不再爬取新数据。")
        return

    excluded_tids = []
    base_url = "https://api.bilibili.com/x/web-interface/newlist?rid={rid}&type=0&ps=25&pn={page}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
        "Referer": "https://www.bilibili.com/",
    }
    cold_videos = []
    all_up_mids = set()
    seen_up_mids = set()

    for page in range(1, max_pages + 1):
        url = base_url.format(rid=rid, page=page)
        print(f"正在请求第 {page} 页：{url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            json_data = response.json()
        except Exception as e:
            print(f"请求第 {page} 页失败: {e}")
            continue

        archives = json_data.get('data', {}).get('archives', [])
        if not archives:
            print("未获取到数据，可能已到最后一页")
            break

        for video in archives:
            tid = video.get('tid', 0)
            if tid in excluded_tids:
                continue

            stat = video.get('stat', {})
            view_count = stat.get('view', 0)
            up_mid = video.get('owner', {}).get('mid', '')

            if up_mid not in seen_up_mids and view_count < view_threshold:
                seen_up_mids.add(up_mid)
                cold_videos.append({
                    'up_mid': up_mid,
                    '视频标题': video.get('title', ''),
                    '视频链接': 'https://www.bilibili.com/video/' + video.get('bvid', ''),
                    'bvid': video.get('bvid', ''),
                    '播放数': view_count,
                    '弹幕数': stat.get('danmaku', 0),
                    '投币数': stat.get('coin', 0),
                    '点赞数': stat.get('like', 0),
                    '分享数': stat.get('share', 0),
                    '收藏数': stat.get('favorite', 0),
                    'cid': video.get('cid', ''),
                    '分区': video.get('tname', ''),
                    '封面': video.get('pic', ''),
                    '时长（秒）': video.get('duration', 0)
                })
                all_up_mids.add(up_mid)
        time.sleep(0.05)

    print(f"开始获取 {len(all_up_mids)} 个UP主的数据...")
    follower_data = get_follower_counts(all_up_mids)

    filtered_videos = []
    for video in cold_videos:
        up_mid = video['up_mid']
        data = follower_data.get(up_mid, {"粉丝数": 0, "历史点赞数": 0, "历史稿件数": 0})
        count = data["粉丝数"]
        like_num = data["历史点赞数"]
        archive_count = data["历史稿件数"]
        p_rating = like_num / archive_count if archive_count > 0 else 0
        video['粉丝数'] = count
        video["历史点赞数"] = like_num
        video["历史稿件数"] = archive_count
        video["p_rating"] = p_rating

        if count < 20 and random.random() < 0.8:
            continue
        elif count < 200 and random.random() < 0.6:
            continue
        filtered_videos.append(video)

        if len(filtered_videos) % 10 == 0:
            df = pd.DataFrame(filtered_videos)
            df = df.drop_duplicates(subset=['up_mid'], keep='first')
            df.to_csv(csv_filepath, index=False, encoding='utf_8_sig')
            print(f"已保存 {len(df)} 条视频数据到 {csv_filepath}")
            filtered_videos = df.to_dict('records')

    if len(filtered_videos) % 10 != 0:
        df = pd.DataFrame(filtered_videos)
        df = df.drop_duplicates(subset=['up_mid'], keep='first')
        df.to_csv(csv_filepath, index=False, encoding='utf_8_sig')
        print(f"已保存全部 {len(df)} 条视频数据到 {csv_filepath}")

    print(f"数据写入成功: {csv_filepath}")


# ------------------------------
# 爬取“popular”数据并写入 CSV 文件
# ------------------------------
def crawl_pop_videos(max_pages=2, view_threshold=100000, csv_filepath=""):
    if os.path.exists(csv_filepath):
        print(f"{csv_filepath} 已存在，不再爬取新数据。")
        return

    base_url = "https://api.bilibili.com/x/web-interface/popular?ps=25&pn={page}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Referer": "https://www.bilibili.com/",
    }
    pop_videos = []
    all_up_mids = set()
    seen_up_mids = set()

    for page in range(1, max_pages + 1):
        url = base_url.format(page=page)
        print(f"正在请求热门视频第 {page} 页：{url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            json_data = response.json()
        except Exception as e:
            print(f"请求第 {page} 页失败: {e}")
            continue

        videos = json_data.get('data', {}).get('list', [])
        if not videos:
            print("未获取到数据，可能已到最后一页")
            break

        for video in videos:
            view_count = video.get('stat', {}).get('vv', 0)  # 使用 vv 字段表示播放量
            up_mid = video.get('owner', {}).get('mid', '')

            if up_mid not in seen_up_mids and view_count >= view_threshold:
                seen_up_mids.add(up_mid)
                pop_videos.append({
                    'up_mid': up_mid,
                    '视频标题': video.get('title', ''),
                    '视频链接': video.get('short_link_v2', 'https://www.bilibili.com/video/' + video.get('bvid', '')),
                    'bvid': video.get('bvid', ''),
                    '播放数': view_count,
                    '弹幕数': video.get('stat', {}).get('danmaku', 0),
                    '投币数': video.get('stat', {}).get('coin', 0),
                    '点赞数': video.get('stat', {}).get('like', 0),
                    '分享数': video.get('stat', {}).get('share', 0),
                    '收藏数': video.get('stat', {}).get('favorite', 0),
                    'cid': video.get('cid', ''),
                    '分区': video.get('tname', ''),
                    '封面': video.get('pic', ''),
                    '时长（秒）': video.get('duration', 0)
                })
                all_up_mids.add(up_mid)
        time.sleep(0.05)

    print(f"开始获取 {len(all_up_mids)} 个UP主的数据...")
    follower_data = get_follower_counts(all_up_mids)

    filtered_videos = []
    for video in pop_videos:
        up_mid = video['up_mid']
        data = follower_data.get(up_mid, {"粉丝数": 0, "历史点赞数": 0, "历史稿件数": 0})
        count = data["粉丝数"]
        like_num = data["历史点赞数"]
        archive_count = data["历史稿件数"]
        p_rating = like_num / archive_count if archive_count > 0 else 0
        video['粉丝数'] = count
        video["历史点赞数"] = like_num
        video["历史稿件数"] = archive_count
        video["p_rating"] = p_rating

        filtered_videos.append(video)

        if len(filtered_videos) % 10 == 0:
            df = pd.DataFrame(filtered_videos)
            df = df.drop_duplicates(subset=['up_mid'], keep='first')
            df.to_csv(csv_filepath, index=False, encoding='utf_8_sig')
            print(f"已保存 {len(df)} 条热门视频数据到 {csv_filepath}")
            filtered_videos = df.to_dict('records')

    if len(filtered_videos) % 10 != 0:
        df = pd.DataFrame(filtered_videos)
        df = df.drop_duplicates(subset=['up_mid'], keep='first')
        df.to_csv(csv_filepath, index=False, encoding='utf_8_sig')
        print(f"已保存全部 {len(df)} 条热门视频数据到 {csv_filepath}")

    print(f"热门视频数据写入成功: {csv_filepath}")


# ------------------------------
# 获取视频详细信息 (unchanged)
# ------------------------------
def fetch_video_info(bvid, cookie):
    url = "https://api.bilibili.com/x/web-interface/view"
    params = {"bvid": bvid}
    cookies = {}
    for item in cookie.split(";"):
        if "=" in item:
            key, val = item.strip().split("=", 1)
            cookies[key] = val
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Referer": "https://www.bilibili.com/"
    }
    try:
        response = requests.get(url, params=params, headers=headers, cookies=cookies, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        if json_data.get("code") != 0:
            return {"code": json_data.get("code"), "message": json_data.get("message", "")}
        else:
            stat = json_data["data"]["stat"]
            return {
                "code": 0,
                "播放数": stat.get("view", 0),
                "弹幕数": stat.get("danmaku", 0),
                "投币数": stat.get("coin", 0),
                "点赞数": stat.get("like", 0),
                "分享数": stat.get("share", 0),
                "收藏数": stat.get("favorite", 0),
                "封面": json_data["data"].get("pic", ""),
                "分区": json_data["data"].get("tname", ""),
                "时长（秒）": json_data["data"].get("duration", 0)
            }
    except Exception as e:
        return {"code": -1, "message": f"请求异常：{e}"}


# ------------------------------
# 更新 CSV 文件并重命名 (unchanged)
# ------------------------------
def update_csv_file(csv_filepath, cookie):
    try:
        df = pd.read_csv(csv_filepath, encoding='utf_8_sig')
    except Exception as e:
        print(f"读取 CSV 文件失败: {e}")
        return False

    default_values = {
        "播放数": 0,
        "弹幕数": 0,
        "投币数": 0,
        "点赞数": 0,
        "分享数": 0,
        "收藏数": 0,
        "封面": "",
        "分区": "",
        "时长（秒）": 0,
        "历史点赞数": 0,
        "历史稿件数": 0,
        "p_rating": 0.0
    }
    for field, default in default_values.items():
        if field not in df.columns:
            df[field] = default

    count = 0
    for index, row in df.iterrows():
        bvid = row.get("bvid")
        if not bvid:
            continue
        result = fetch_video_info(bvid, cookie)
        if result.get("code") == 0:
            df.at[index, "播放数"] = result.get("播放数", 0)
            df.at[index, "弹幕数"] = result.get("弹幕数", 0)
            df.at[index, "投币数"] = result.get("投币数", 0)
            df.at[index, "点赞数"] = result.get("点赞数", 0)
            df.at[index, "分享数"] = result.get("分享数", 0)
            df.at[index, "收藏数"] = result.get("收藏数", 0)
            df.at[index, "封面"] = result.get("封面", "")
            df.at[index, "分区"] = result.get("分区", "")
            df.at[index, "时长（秒）"] = result.get("时长（秒）", 0)
            print(f"视频 {bvid} 更新成功。")
        else:
            print(f"视频 {bvid} 更新失败: code={result.get('code')}, message={result.get('message')}")
        time.sleep(0.1)
        count += 1
        if count % 10 == 0:
            df.to_csv(csv_filepath, index=False, encoding='utf_8_sig')
            print(f"已保存更新后的 {count} 条视频数据到 {csv_filepath}")

    if count % 10 != 0 or count > 0:
        df.to_csv(csv_filepath, index=False, encoding='utf_8_sig')
        print(f"CSV 文件更新完毕: {csv_filepath}")

    if count > 0:
        new_filepath = csv_filepath.replace(".csv", "_updated.csv")
        os.rename(csv_filepath, new_filepath)
        print(f"文件已重命名为: {new_filepath}")
        return True
    return False


# ------------------------------
# 主流程
# ------------------------------
def main():
    config, data_dir, recrawl_days = load_or_create_config("config.properties")
    if config is None:
        return

    try:
        cookie = config.get("bilibili", "cookie")
        print(f"数据目录: {data_dir}, 更新时间阈值: {recrawl_days} 天")
    except Exception as e:
        print(f"读取配置项错误: {e}")
        return

    threshold_time = datetime.now() - timedelta(days=recrawl_days)

    print(f"开始检查 {data_dir} 目录下需要更新的 CSV 文件...")
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and "_updated" not in filename:
            csv_filepath = os.path.join(data_dir, filename)
            try:
                time_str = filename.split("videoList_")[1].split(".csv")[0]
                file_time = datetime.strptime(time_str, "%Y%m%d_%H%M")
                if file_time < threshold_time:
                    print(f"正在更新 {csv_filepath}（文件时间: {file_time}）...")
                    update_csv_file(csv_filepath, cookie)
                else:
                    print(f"跳过 {csv_filepath}（文件时间: {file_time} 未超过 {recrawl_days} 天）")
            except (IndexError, ValueError) as e:
                print(f"无法解析 {filename} 的时间戳: {e}，跳过更新")

    # 创建新视频和热门视频的 CSV 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # 爬取新视频
    new_csv_filename = f"videoList_new_{timestamp}.csv"
    new_csv_filepath = os.path.join(data_dir, new_csv_filename)
    print(f"开始爬取新视频数据到 {new_csv_filepath}...")
    crawl_new_videos(rid=0, max_pages=6, view_threshold=10000, csv_filepath=new_csv_filepath)
    print("新视频数据已爬取并保存到 CSV 文件中。")

    # 爬取热门视频
    pop_csv_filename = f"videoList_pop_{timestamp}.csv"
    pop_csv_filepath = os.path.join(data_dir, pop_csv_filename)
    print(f"开始爬取热门视频数据到 {pop_csv_filepath}...")
    crawl_pop_videos(max_pages=1, view_threshold=100000, csv_filepath=pop_csv_filepath)
    print("热门视频数据已爬取并保存到 CSV 文件中。")


if __name__ == '__main__':
    main()
