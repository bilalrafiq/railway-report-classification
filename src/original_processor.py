import pandas as pd
import numpy as np
import json
import tqdm
import math
import os
import re
from collections import Counter

# ==========================================
# 有文本增强不全，
# ==========================================
def clean_text_for_plm(text):
    """标点符号前后补空格，彻底去除 nan"""
    if not isinstance(text, str): return ""
    text = re.sub(r'([,.!?()])', r' \1 ', text)
    # 去除多余空格和 nan 字符串
    text = " ".join([p for p in text.split() if p.lower() != 'nan' and p.upper() != 'NONE'])
    return text.strip()

def get_weather(code):
    mapping = {"1": "clear", "2": "cloudy", "3": "rain", "4": "fog", "5": "sleet"}
    return mapping.get(str(code).strip(), "snow")

def get_Type_accident(code):
    mapping = {
        "1": "Derailment", "2": "Head on collision", "3": "Rearend collision",
        "4": "Side collision", "5": "Raking collision", "6": "Broken train collision",
        "7": "Hwy-rail crossing narr", "8": "RR Grade Crossing", "9": "Obstruction",
        "10": "Explosive-detonation", "11": "fire/violent rupture"
    }
    return mapping.get(str(code).strip(), "Other impacts")

def get_VISIBLTY(code):
    mapping = {"1": "dawn", "2": "day", "3": "dusk"}
    return mapping.get(str(code).strip(), "dark")

# ==========================================
# 2. 时间特征处理 (按您提供的逻辑)
# ==========================================
def process_time_features(row):
    """
    输入: 原始的一行数据 row
    输出: 
      1. time_feats: 一个字典，包含处理好的数值型时间特征 (给结构化模型用)
      2. time_text: 一个字符串，包含人类可读的时间描述 (给文本模型用)
    """
    # 1. 提取原始值
    try:
        year = int(float(row.get('YEAR4', 0) or 0))
        month = int(float(row.get('MONTH', 0) or 1))
        day = int(float(row.get('DAY', 0) or 1))
        hour = int(float(row.get('TIMEHR', 0) or 0))
        minute = int(float(row.get('TIMEMIN', 0) or 0))
        ampm = row.get('AMPM', '').strip().upper()
    except:
        # 如果出错，返回默认值
        return {}, ""

    # 2. 处理小时 (转换成 24小时制)
    # 逻辑: 12 AM -> 0, 1 AM -> 1 ... 11 AM -> 11
    #       12 PM -> 12, 1 PM -> 13 ... 11 PM -> 23
    if ampm == 'PM' and hour != 12:
        hour += 12
    elif ampm == 'AM' and hour == 12:
        hour = 0
    
    # 3. 生成结构化特征 (Cyclical Encoding)
    # 这里的原理是把时间映射到单位圆上
    time_feats = {}
    
    # 月份 (1-12) -> 周期
    time_feats['Month_Sin'] = math.sin(2 * math.pi * month / 12)
    time_feats['Month_Cos'] = math.cos(2 * math.pi * month / 12)
    
    # 小时 (0-23) + 分钟 -> 精确的周期
    # 把分钟也算进去，变成小数小时 (例如 6:30 -> 6.5)
    exact_hour = hour + (minute / 60.0)
    time_feats['Hour_Sin'] = math.sin(2 * math.pi * exact_hour / 24)
    time_feats['Hour_Cos'] = math.cos(2 * math.pi * exact_hour / 24)
    
    # 年份 (做简单的归一化，假设数据在 1975-2025 之间)
    # 这样把年份变成 0-1 之间的数，对神经网络更友好
    time_feats['Year_Norm'] = (year - 1975) / 50.0 

    # 4. 生成文本描述
    # 补零操作，例如 6:1 变成 06:01
    time_str = f"{hour:02d}:{minute:02d}"
    date_str = f"{year}-{month:02d}-{day:02d}"
    time_text = f"Time: {date_str} {time_str}."

    return time_feats, time_text


# def process_time_features(row):
#     try:
#         year = int(float(row.get('YEAR4', 0) or 0))
#         month = int(float(row.get('MONTH', 0) or 1))
#         day = int(float(row.get('DAY', 0) or 1))
#         hour = int(float(row.get('TIMEHR', 0) or 0))
#         minute = int(float(row.get('TIMEMIN', 0) or 0))
#         ampm = str(row.get('AMPM', '')).strip().upper()
#     except:
#         return {}, ""

#     if ampm == 'PM' and hour != 12:
#         hour += 12
#     elif ampm == 'AM' and hour == 12:
#         hour = 0
    
#     time_feats = {}
#     time_feats['Month_Sin'] = math.sin(2 * math.pi * month / 12)
#     time_feats['Month_Cos'] = math.cos(2 * math.pi * month / 12)
#     exact_hour = hour + (minute / 60.0)
#     time_feats['Hour_Sin'] = math.sin(2 * math.pi * exact_hour / 24)
#     time_feats['Hour_Cos'] = math.cos(2 * math.pi * exact_hour / 24)
#     time_feats['Year_Norm'] = (year - 1975) / 50.0 

#     date_text = f"{year}.{month:02d}.{day:02d} {ampm}"
#     return time_feats, date_text

# ==========================================
# 3. 核心处理器
# ==========================================
USD_TO_CNY_RATES = {
    '2025': 7.20, '2024': 7.19, '2023': 7.05, '2022': 6.73,
    '2021': 6.45, '2020': 6.90, '2019': 6.90, '2018': 6.62,
    '2017': 6.75, '2016': 6.64, 'default': 7.00
}

class FinalProcessorV6:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        # 数值列 (10路原始 + 时间)
        self.num_cols = ['TEMP', 'TRNSPD', 'HIGHSPD', 'TONS', 'CARS', 'CARSDMG', 'CARSHZD', 'Latitude', 'Longitud', 'TRKDNSTY']
        # 类别列 (包含 V6 新增列)
        self.cat_cols = ['TYPE', 'STATE', 'RAILROAD', 'COUNTY', 'TRNNBR', 'VISIBLTY', 'WEATHER', 'TRKCLAS', 'TYPTRK', 'TYPRR', 'PASSTRN', 'SIGNAL', 'MOPERA', 'TRNDIR', 'STATION', 'TRKNAME']

    def _clean_num(self, val):
        if pd.isna(val) or str(val).strip().lower() in ["", "nan", "."]: return 0.0
        try: return float(str(val).replace(',', ''))
        except: return 0.0

    def _calculate_severity(self, row):
        """采用提供的 0-5 级行政定级逻辑"""
        killed = int(self._clean_num(row.get('CASKLD')))
        injured = int(self._clean_num(row.get('CASINJ')))
        loss_usd = self._clean_num(row.get('ACCDMG'))
        
        year_str = str(row.get('YEAR4', '2020'))
        rate = USD_TO_CNY_RATES.get(year_str, USD_TO_CNY_RATES['default'])
        loss_cny = loss_usd * rate
        
        p_derail = int(self._clean_num(row.get('LOADP2')) + self._clean_num(row.get('EMPTYP2')))
        f_derail = int(self._clean_num(row.get('LOADF2')) + self._clean_num(row.get('EMPTYF2')) + self._clean_num(row.get('CABOOSE2')))

        if killed >= 30 or injured >= 100 or loss_cny >= 100000000:
            return 5, "特别重大事故"
        elif (10 <= killed <= 29) or (50 <= injured <= 99) or (50000000 <= loss_cny < 100000000) or (p_derail >= 18) or (f_derail >= 60):
            return 4, "重大事故"
        elif (3 <= killed <= 9) or (10 <= injured <= 49) or (10000000 <= loss_cny < 50000000) or (2 <= p_derail <= 17) or (6 <= f_derail <= 59):
            return 3, "较大事故"
        elif (killed == 2) or (5 <= injured <= 9) or (5000000 <= loss_cny < 10000000) or (p_derail == 1) or (4 <= f_derail <= 5):
            return 2, "一般A类事故"
        elif (killed == 1) or (1 <= injured <= 4) or (1000000 <= loss_cny < 5000000) or (2 <= f_derail <= 3):
            return 1, "一般B类事故"
        else:
            return 0, "一般C类事故"

    def process(self):
        df = pd.read_csv(self.input_path, low_memory=False, dtype=str)
        results = []
        seen_ids = set()

        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="6分类标签式增强"):
            # 1. 清洗与去重
            if str(row.get('NARRLEN', '0')).strip() in ['0', '', 'nan']: continue
            uid = f"{row.get('RAILROAD')}_{row.get('INCDTNO')}_{row.get('YEAR')}"
            if uid in seen_ids: continue
            seen_ids.add(uid)

            # 2. 标签逻辑 (行政 6 分类)
            label_id, label_desc = self._calculate_severity(row)

            # 3. 时间处理
            time_feats, time_text = process_time_features(row)

            # 4. 数值与类别特征构建
            num_data = {f: self._clean_num(row.get(f)) for f in self.num_cols}
            num_data.update(time_feats)
            num_data['Kinetic_Energy'] = 0.5 * num_data['TONS'] * (num_data['TRNSPD'] ** 2)
            
            cat_data = {f: str(row.get(f, 'UNKNOWN')).strip() for f in self.cat_cols}

            # 5. 文本流：标签化增强格式
            weather_word = get_weather(row.get('WEATHER'))
            vis_word = get_VISIBLTY(row.get('VISIBLTY'))
            type_word = get_Type_accident(row.get('TYPE'))

            narr_parts = [str(row.get(f'NARR{i}', '')).strip() for i in range(1, 16)]
            narrative = clean_text_for_plm(" ".join(narr_parts))

            # 🚨 新的文本拼接逻辑（强烈建议）
            text_input = (
                f"{time_text} "
                # f"Location: {cat_data['COUNTY']}, {cat_data['STATE']}. "
                # f"Station: {cat_data['STATION']}. Track: {cat_data['TRKNAME']}. "
                f"Visibility: {vis_word}. Weather: {weather_word}. "
                # f"Train speed {num_data['TRNSPD']} mph, "
                # f"High speed limit {num_data['HIGHSPD']} mph. "
                # f"Cars involved {num_data['CARS']}. "
                f"Accident type: {type_word}. "
                f"Narrative: {narrative}"
            )

            # 6. 保留详细指标
            severity_metrics = {
                "Killed_Count": int(self._clean_num(row.get('CASKLD'))),
                "Injured_Count": int(self._clean_num(row.get('CASINJ'))),
                "Damage_USD": self._clean_num(row.get('ACCDMG'))
            }

            results.append({
                "label": label_id,
                "label_desc": label_desc,
                "severity_metrics": severity_metrics,
                "structured_data": {
                    "numeric": num_data,
                    "categorical": cat_data
                },
                "text_data": text_input
            })

        # 保存
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ 处理完成！生成行政6分类带Tag增强数据。路径: {self.output_path}")

if __name__ == "__main__":
    processor = FinalProcessorV6("../Data/datanew.csv", "../Data/new/0201/dataset_final_v6_admin4.jsonl")
    processor.process()