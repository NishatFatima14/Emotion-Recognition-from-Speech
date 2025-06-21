# 🎙️ Emotion Recognition from Speech (CodeAlpha Internship Project)

This project uses deep learning and speech processing to classify emotions from audio files.

## 🔧 Technologies Used
- Python
- Librosa
- TensorFlow / Keras
- Scikit-learn
- Matplotlib / Seaborn

## 📁 Dataset
Used the RAVDESS Emotional Speech Audio dataset (1440 files, 8 emotions):  
- Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised

## 🧠 Model
- Extracted MFCC features (40 per audio)
- Built a CNN model with:
  - 2 Conv1D layers
  - MaxPooling & Dropout
  - Fully Connected Softmax output
- Final test accuracy: **✅ 87.6%** (example)
- Confusion matrix & classification report included

