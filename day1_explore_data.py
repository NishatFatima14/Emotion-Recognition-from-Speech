import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Path to your RAVDESS folder
DATA_PATH = "D:/Emotion Recognition from Speech/ravdess_data"

# Function to extract emotion label from filename
def get_emotion_from_filename(filename):
    emotion_code = int(filename.split("-")[2])
    emotions = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    return emotions[emotion_code]

# Pick a sample file to test
sample_actor_folder = os.path.join(DATA_PATH, "Actor_04")
sample_file = os.listdir(sample_actor_folder)[4]
file_path = os.path.join(sample_actor_folder, sample_file)

# for sample_file in os.listdir(sample_actor_folder):
#     emotion = get_emotion_from_filename(sample_file)
#     print(f"{sample_file} → {emotion}")


# Load the audio
signal, sr = librosa.load(file_path)

# Show info
print("File Name:", sample_file)
print("Emotion Label:", get_emotion_from_filename(sample_file))
print("Signal Shape:", signal.shape)
print("Sampling Rate:", sr)

# Plot waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(signal, sr=sr)
plt.title(f"Waveform - {get_emotion_from_filename(sample_file)}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
