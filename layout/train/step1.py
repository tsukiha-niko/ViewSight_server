import glob
import os
import random
from datetime import datetime

import pandas as pd

# 修改为你的CSV目录路径（可以是绝对路径）
csv_dir = r"C:\Users\ElmCose\Desktop\打印\temp"
os.chdir(csv_dir)

csv_files = glob.glob("*.csv")

if not csv_files:
    print("❌ 没有找到任何 CSV 文件")
else:
    all_dfs = []
    # 获取每个文件的修改时间
    file_info = []
    for file in csv_files:
        mod_time = os.path.getmtime(file)
        file_info.append({'file': file, 'mod_time': mod_time})

    # 按修改时间排序，最新的排在最后（这样concat后最新的记录会覆盖旧的）
    file_info.sort(key=lambda x: x['mod_time'])

    for info in file_info:
        file = info['file']
        df = pd.read_csv(file)
        df['源文件'] = os.path.basename(file)
        df['文件修改时间'] = datetime.fromtimestamp(info['mod_time'])
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)

    # 删除"视频链接"列中的重复数据，保留最后一条（最新修改的）
    merged_df = merged_df.drop_duplicates(subset=['视频链接'], keep='last')

    # 删除 p_rating 为 0 的行
    merged_df = merged_df[merged_df['p_rating'] != 0]


    # 删除播放数相关行
    def filter_by_play_count(row):
        play_count = row.get('播放数', 0)
        if play_count == 0:
            return random.random() > 0.8  # 80% 概率删除
        elif play_count < 10:
            return random.random() > 0.5  # 50% 概率删除
        elif play_count < 100:
            return random.random() > 0.3  # 30% 概率删除
        elif play_count < 1000:
            return random.random() > 0.1  # 10% 概率删除
        else:
            return True  # 播放数大于等于1000时保留


    # 应用删除过滤函数
    merged_df = merged_df[merged_df.apply(filter_by_play_count, axis=1)]

    # 删除临时添加的修改时间列（如果需要保留可以注释掉这行）
    merged_df = merged_df.drop(columns=['文件修改时间'])

    merged_df.to_csv("merged.csv", index=False, encoding='utf-8-sig')
    print("✅ 合并完成，重复视频链接已删除，播放数相关行已删除，文件已保存为 merged.csv")
