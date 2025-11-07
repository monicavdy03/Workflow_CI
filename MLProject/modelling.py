import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import joblib
import os

data_path = r"C:\Users\monica\Downloads\SMSML_Monika Dian Vidya Putri\Eksperimen_SML_Monika Dian Vidya Putri\preprocessing\brain_preprocessing\brain_data.pkl"
X_train, X_test, y_train, y_test = joblib.load(data_path)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import os
tracking_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mlruns"))
mlflow.set_tracking_uri(f"file:///{tracking_dir}")

mlflow.set_experiment("Brain_Tumor_Classification")

mlflow.keras.autolog()

with mlflow.start_run(run_name="cnn_brain_tumor_basic"):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f" Akurasi Uji: {test_acc:.4f}")
