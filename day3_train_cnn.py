import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("mfcc_features.csv")

# Separate features and labels
X = df.drop("label", axis=1).values
y = df["label"].values

# Encode emotion labels into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape input for CNN: [samples, timesteps, features=1]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42)

model = Sequential()

model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=(40, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv1D(128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='softmax'))  # 8 emotion classes

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)
loss, accuracy = model.evaluate(X_test, y_test)
print("🎯 Test Accuracy:", round(accuracy * 100, 2), "%")

# plot Accuracy (Optional but Useful)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# save the model
model.save("emotion_cnn_model.h5")
print("✅ Model saved as emotion_cnn_model.h5")





































# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
#
# # Load the CSV created in Day 2
# df = pd.read_csv("mfcc_features.csv")
#
# # Separate features and labels
# X = df.drop("label", axis=1).values
# y = df["label"].values
#
# # Encode the emotion labels into numbers
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# y_categorical = to_categorical(y_encoded)
#
# # CNN expects 3D input: [samples, time steps, features]
# X = X.reshape(X.shape[0], X.shape[1], 1)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_categorical, test_size=0.2, random_state=42)
#
# model = Sequential()
#
# model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=(40, 1)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.3))
#
# model.add(Conv1D(128, kernel_size=5, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.3))
#
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(8, activation='softmax'))  # 8 emotion classes
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
#
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print("🎯 Test Accuracy:", round(test_accuracy * 100, 2), "%")
#
# import matplotlib.pyplot as plt
#
# plt.plot(history.history['accuracy'], label='Train Acc')
# plt.plot(history.history['val_accuracy'], label='Val Acc')
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training vs Validation Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()
#
