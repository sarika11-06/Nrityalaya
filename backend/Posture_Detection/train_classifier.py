import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
with open('pose_data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.array(data_dict['data'])  # shape: (n_samples, n_features like 66, 84, etc.)
labels = np.array(data_dict['labels'])

# Encode labels (e.g., 'Alapadma', 'Pataka', etc.)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Convert labels to categorical (one-hot encoding)
labels_categorical = to_categorical(labels_encoded, num_classes)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_categorical, test_size=0.2, random_state=42, stratify=labels)

# ✅ Dynamically get input feature size
input_shape = x_train.shape[1]

# Build the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(input_shape,)))  # ✅ Correct input shape
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"✅ Test accuracy: {test_acc * 100:.2f}%")

# Save the model and label encoder
model.save('cnn_model.h5')
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Model and label encoder saved.")
