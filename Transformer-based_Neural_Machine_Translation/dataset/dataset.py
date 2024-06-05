import pandas as pd
import re

# Load the CSV file
file_path = 'train.csv'
df = pd.read_csv(file_path)

# Function to detect if the sentence is English
def is_english(sentence):
    # Check if the sentence contains Chinese characters
    return not bool(re.search('[\u4e00-\u9fff]', sentence))

# Separate English and Chinese sentences
english_sentences = []
chinese_sentences = []

for src, tgt in zip(df['src'], df['tgt']):
    if is_english(src):
        english_sentences.append(src)
        chinese_sentences.append(tgt)
    else:
        english_sentences.append(tgt)
        chinese_sentences.append(src)

# Save to text files
with open('train_en.txt', 'w', encoding='utf-8') as src_file:
    src_file.write('\n'.join(english_sentences))

with open('train_zh.txt', 'w', encoding='utf-8') as tgt_file:
    tgt_file.write('\n'.join(chinese_sentences))

print("Files generated successfully")
