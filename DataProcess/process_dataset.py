import pandas as pd
import json
from collections import defaultdict, Counter

# 读取CSV文件
print("Reading CSV file...")
df = pd.read_csv('go_emotions_dataset.csv')

# 获取所有情感标签列（排除id、text和example_very_unclear列）
emotion_columns = [col for col in df.columns if col not in ['id', 'text', 'example_very_unclear']]

# 创建一个字典来存储文本和对应的所有情感标签
text_emotions = defaultdict(list)

print("Processing data and counting emotion occurrences...")
# 处理每一行数据
for _, row in df.iterrows():
    text = row['text']
    
    # 获取标记为1的情感标签
    emotions = [emotion for emotion in emotion_columns if row[emotion] == 1]
    
    # 如果example_very_unclear为True，添加unclear标签
    if row['example_very_unclear']:
        emotions.append('unclear')
    
    # 将这行的情感标签添加到对应文本的列表中
    text_emotions[text].extend(emotions)

# 转换为最终的格式，计算每个标签的出现次数
processed_data = []
for text, emotions in text_emotions.items():
    # 使用Counter计算每个标签出现的次数
    emotion_counts = Counter(emotions)
    
    # 转换为所需的格式：[{tag: "emotion", cnt: count}, ...]
    emotion_label = [
        {"tag": emotion, "cnt": count}
        for emotion, count in emotion_counts.items()
    ]
    
    # 按出现次数降序排序
    emotion_label.sort(key=lambda x: (-x["cnt"], x["tag"]))
    
    processed_data.append({
        "text": text,
        "emotion_label": emotion_label
    })

# 打印去重统计
print("\nDeduplication statistics:")
print(f"Original number of records: {len(df)}")
print(f"Number of unique texts: {len(processed_data)}")
print(f"Removed {len(df) - len(processed_data)} duplicate texts")

# 计算标签统计信息
all_counts = []
for item in processed_data:
    counts = [emotion["cnt"] for emotion in item["emotion_label"]]
    all_counts.extend(counts)

avg_count = sum(all_counts) / len(all_counts)
max_count = max(all_counts)
min_count = min(all_counts)

print("\nEmotion count statistics:")
print(f"Average occurrence count per emotion: {avg_count:.2f}")
print(f"Maximum occurrence count: {max_count}")
print(f"Minimum occurrence count: {min_count}")

# 计算每个文本的不同情感标签数量
emotion_variety = [len(item["emotion_label"]) for item in processed_data]
avg_emotions = sum(emotion_variety) / len(emotion_variety)
max_emotions = max(emotion_variety)

print("\nEmotion variety statistics:")
print(f"Average number of different emotions per text: {avg_emotions:.2f}")
print(f"Maximum number of different emotions for a text: {max_emotions}")

# 将处理后的数据保存为JSON文件
print("\nSaving processed data...")
with open('processed_emotions_with_counts.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

# 显示一些样例，特别是具有多个情感标签的例子
print("\nSample records:")
# 按情感标签数量排序，显示一些具有不同数量标签的例子
samples = sorted(processed_data, key=lambda x: len(x["emotion_label"]), reverse=True)[:5]
for i, record in enumerate(samples):
    print(f"\nRecord {i+1} (with {len(record['emotion_label'])} different emotions):")
    print(json.dumps(record, ensure_ascii=False, indent=2))

print("\nProcessing completed! Data saved to 'processed_emotions_with_counts.json'") 