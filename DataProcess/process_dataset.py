import pandas as pd
import json
from collections import defaultdict, Counter

# read the csv file
print("Reading CSV file...")
df = pd.read_csv('go_emotions_dataset.csv')

# get all emotion columns (exclude id, text and example_very_unclear columns)
emotion_columns = [col for col in df.columns if col not in ['id', 'text', 'example_very_unclear']]

# create a dictionary to store text and all emotion labels
text_emotions = defaultdict(list)

print("Processing data and counting emotion occurrences...")
for _, row in df.iterrows():
    text = row['text']
    
    # get the emotion labels marked as 1
    emotions = [emotion for emotion in emotion_columns if row[emotion] == 1]
    
    # if example_very_unclear is True, add unclear label
    if row['example_very_unclear']:
        emotions.append('unclear')
    
    # add the emotion labels to the corresponding text list
    text_emotions[text].extend(emotions)

# convert to the final format, calculate the occurrence of each label
processed_data = []
for text, emotions in text_emotions.items():
    # use Counter to calculate the occurrence of each label
    emotion_counts = Counter(emotions)
    
    # convert to the required format: [{tag: "emotion", cnt: count}, ...]
    emotion_label = [
        {"tag": emotion, "cnt": count}
        for emotion, count in emotion_counts.items()
    ]
    
    # sort by occurrence count in descending order
    emotion_label.sort(key=lambda x: (-x["cnt"], x["tag"]))
    
    processed_data.append({
        "text": text,
        "emotion_label": emotion_label
    })

# print deduplication statistics
print("\nDeduplication statistics:")
print(f"Original number of records: {len(df)}")
print(f"Number of unique texts: {len(processed_data)}")
print(f"Removed {len(df) - len(processed_data)} duplicate texts")

# calculate label statistics
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

# calculate the number of different emotion labels for each text
emotion_variety = [len(item["emotion_label"]) for item in processed_data]
avg_emotions = sum(emotion_variety) / len(emotion_variety)
max_emotions = max(emotion_variety)

print("\nEmotion variety statistics:")
print(f"Average number of different emotions per text: {avg_emotions:.2f}")
print(f"Maximum number of different emotions for a text: {max_emotions}")

# save the processed data as a JSON file
print("\nSaving processed data...")
with open('processed_emotions_with_counts.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

# display some examples, especially examples with multiple emotion labels
print("\nSample records:")
# sort by the number of emotion labels, display some examples with different numbers of labels
samples = sorted(processed_data, key=lambda x: len(x["emotion_label"]), reverse=True)[:5]
for i, record in enumerate(samples):
    print(f"\nRecord {i+1} (with {len(record['emotion_label'])} different emotions):")
    print(json.dumps(record, ensure_ascii=False, indent=2))

print("\nProcessing completed! Data saved to 'processed_emotions_with_counts.json'") 