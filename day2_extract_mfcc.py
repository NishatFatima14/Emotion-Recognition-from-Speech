import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_PATH = r"C:\Users\HP\PycharmProjects\Emotion_Recognition_from_Speech\Emotion Recognition from Speech\ravdess_data" # <<< UPDATE THIS TO YOUR REAL PATH

emotion_map = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

def extract_emotion(filename):
    try:
        return emotion_map[int(filename.split("-")[2])]
    except:
        return None

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"❌ Error with {file_path}: {e}")
        return None

features = []
labels = []

print("🔍 DATA_PATH points to:", DATA_PATH)
print("🔍 Folder contains:", os.listdir(DATA_PATH))

# -----------------------------

# print("🚀 Starting feature extraction...\n")
# for actor_folder in tqdm(os.listdir(DATA_PATH)):


print("🚀 Starting feature extraction...\n")
for actor_folder in tqdm(os.listdir(DATA_PATH)):
    actor_path = os.path.join(DATA_PATH, actor_folder)
    if not os.path.isdir(actor_path):
        continue
    for filename in os.listdir(actor_path):
        file_path = os.path.join(actor_path, filename)
        label = extract_emotion(filename)
        if label is None:
            continue
        mfcc_features = extract_features(file_path)
        if mfcc_features is not None:
            features.append(mfcc_features)
            labels.append(label)
            print(f"✅ Processed: {filename} → {label}")

print("\n✅ Total processed files:", len(features))
print("🧪 Labels found:", set(labels))

# Final CSV write
if len(features) > 0:
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv("mfcc_features.csv", index=False)
    print("🎉 Features saved to mfcc_features.csv")
else:
    print("⚠️ No data extracted. Please check your folder path or file structure.")
