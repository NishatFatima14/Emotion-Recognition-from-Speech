# ✅ Step 1: Import What We Need
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Step 2: Load the Data Again
df = pd.read_csv(r"C:\Users\HP\PycharmProjects\Emotion_Recognition_from_Speech\mfcc_features.csv")


X = df.drop("label", axis=1).values
y = df["label"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-hot for model
y_categorical = pd.get_dummies(y).values

# Reshape X for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42)

y_test_labels = np.argmax(y_test, axis=1)  # Get original labels back

# ✅ Step 3: Load Your Trained Model
model = load_model("emotion_cnn_model.h5")

# ✅ Step 4: Make Predictions
y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)  # Convert one-hot to label index

# ✅ Step 5: Confusion Matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ✅ Step 6: Classification Report
report = classification_report(y_test_labels, y_pred_labels, target_names=le.classes_)
print("📄 Classification Report:\n")
print(report)
