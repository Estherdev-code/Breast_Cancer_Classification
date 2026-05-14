"""
app.py
------
Flask Web Application — Breast Cancer Detection System
Loads the trained ResNet50 weights and serves diagnostic predictions.

Run: python app.py
Then open: http://127.0.0.1:5000
"""

import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from preprocess import preprocess_from_array, prepare_for_model
import sqlite3
from datetime import datetime

app = Flask(__name__)




# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
WEIGHTS_PATH = "resnet50_weights.weights.h5"
DATABASE     = "diagnostics.db"

# ─────────────────────────────────────────────
# DOWNLOAD MODEL FROM HUGGING FACE IF NOT PRESENT
# ─────────────────────────────────────────────
if not os.path.exists(WEIGHTS_PATH):
    print("Downloading model from Hugging Face...")
    from huggingface_hub import hf_hub_download
    downloaded_path = hf_hub_download(
        repo_id="Estherdev-code/breast-cancer-resnet50",
        filename="resnet50_weights.weights.h5"
    )
    # Copy/move file to expected path
    import shutil
    shutil.copy(downloaded_path, WEIGHTS_PATH)
    print("Model downloaded successfully.")
def build_model():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    return Model(inputs=base_model.input, outputs=output)


print("Building model architecture...")
model = build_model()
print("Loading trained weights...")
model.load_weights(WEIGHTS_PATH)
print("Model loaded successfully.")


# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnostics (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            filename   TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_result(filename, prediction, confidence):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO diagnostics (filename, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (filename, prediction, round(confidence, 4), datetime.now().isoformat()))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_image(image_array):
    processed = preprocess_from_array(image_array)
    model_input = prepare_for_model(processed)
    probability = float(model.predict(model_input, verbose=0)[0][0])

    if probability >= 0.5:
        label = "MALIGNANT"
        confidence = probability * 100
    else:
        label = "BENIGN"
        confidence = (1 - probability) * 100

    return label, round(confidence, 2)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    label, confidence = predict_image(image)
    save_result(file.filename, label, confidence)

    return jsonify({
        "filename":   file.filename,
        "prediction": label,
        "confidence": confidence,
        "message":    f"The lesion appears {label.lower()} with {confidence}% confidence.",
        "disclaimer": "This result is AI-generated and must be reviewed by a qualified radiologist."
    })


@app.route("/history")
def history():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM diagnostics ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    return jsonify([
        {"id": r[0], "filename": r[1], "prediction": r[2],
         "confidence": r[3], "timestamp": r[4]}
        for r in rows
    ])


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
init_db()  # initialise database on import so gunicorn picks it up

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
