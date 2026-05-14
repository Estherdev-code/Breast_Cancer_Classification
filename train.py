"""
train.py
--------
Stage 2: Diagnostic Intelligence Layer
Builds, trains, and saves the ResNet50-based breast cancer classifier.
Uses dicom_info.csv to correctly resolve image paths.

Run: python train.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from preprocess import preprocess

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MASS_CSV      = "cbis-ddsm/csv/mass_case_description_train_set.csv"
DICOM_CSV     = "cbis-ddsm/csv/dicom_info.csv"
IMAGE_ROOT    = "cbis-ddsm/jpeg"
MODEL_SAVE    = "breast_cancer_model.h5"
BATCH_SIZE    = 16
EPOCHS        = 30
LEARNING_RATE = 1e-4


# ─────────────────────────────────────────────
# STEP 1: LOAD AND MATCH IMAGES TO LABELS
# ─────────────────────────────────────────────
def load_cbis_ddsm(mass_csv, dicom_csv, image_root):
    """
    Merges mass_case_description_train_set.csv with dicom_info.csv
    to correctly match each full mammogram image to its label.
    Labels: MALIGNANT = 1, BENIGN = 0
    """
    mass_df  = pd.read_csv(mass_csv)
    dicom_df = pd.read_csv(dicom_csv)

    # Keep only full mammogram images
    full_mammo = dicom_df[dicom_df["SeriesDescription"] == "full mammogram images"].copy()

    # Fix path prefix to match local folder
    def resolve_folder(image_path):
        p = str(image_path).strip()
        p = p.replace("CBIS-DDSM/jpeg/", "").replace("CBIS-DDSM\\jpeg\\", "")
        # Remove filename portion if present
        parts = p.replace("\\", "/").split("/")
        # Keep only the numeric folder parts
        folder_parts = [x for x in parts if x and not x.endswith(".dcm") and not x.endswith(".jpg")]
        return os.path.join(image_root, *folder_parts)

    full_mammo["local_folder"] = full_mammo["image_path"].apply(resolve_folder)

    # Find the actual jpg file inside each folder
    def find_jpg(folder_path):
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.lower().endswith(".jpg"):
                    return os.path.join(folder_path, f)
        return None

    full_mammo["image_file"] = full_mammo["local_folder"].apply(find_jpg)
    full_mammo = full_mammo.dropna(subset=["image_file"])

    # Clean PatientID: remove trailing _1, _2 suffix
    full_mammo["clean_id"] = full_mammo["PatientID"].apply(
        lambda x: "_".join(str(x).strip().split("_")[1:3])
    )

    # Prepare labels
    mass_df["label"] = mass_df["pathology"].apply(
        lambda x: 1 if str(x).strip().upper() == "MALIGNANT" else 0
    )
    mass_df["clean_id"] = mass_df["patient_id"].apply(lambda x: str(x).strip())

    # Merge
    merged = pd.merge(
        full_mammo[["image_file", "clean_id"]],
        mass_df[["clean_id", "label"]],
        on="clean_id",
        how="inner"
    ).drop_duplicates(subset=["image_file"])

    print(f"\nTotal matched samples: {len(merged)}")
    print(f"  Malignant: {merged['label'].sum()}")
    print(f"  Benign:    {(merged['label'] == 0).sum()}")

    if len(merged) == 0:
        print("\nDEBUG - Sample folder path:", full_mammo["local_folder"].iloc[0] if len(full_mammo) > 0 else "none")
        print("DEBUG - Sample image_file:", full_mammo["image_file"].iloc[0] if len(full_mammo) > 0 else "none")
        raise ValueError("No images matched. Check folder structure.")

    return merged["image_file"].tolist(), merged["label"].tolist()


# ─────────────────────────────────────────────
# STEP 2: DATA GENERATOR
# ─────────────────────────────────────────────
class MammogramGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, augment=False):
        self.paths      = list(paths)
        self.labels     = list(labels)
        self.batch_size = batch_size
        self.augment    = augment

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths  = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        for path in batch_paths:
            try:
                img = preprocess(path)
                img = img.astype("float32") / 255.0
                if self.augment:
                    if np.random.rand() > 0.5:
                        img = np.fliplr(img)
                    if np.random.rand() > 0.5:
                        img = np.flipud(img)
                images.append(img)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                images.append(np.zeros((224, 224, 3), dtype="float32"))

        return np.array(images), np.array(batch_labels, dtype="float32")

    def on_epoch_end(self):
        combined = list(zip(self.paths, self.labels))
        np.random.shuffle(combined)
        self.paths, self.labels = zip(*combined)


# ─────────────────────────────────────────────
# STEP 3: BUILD THE RESNET50 MODEL
# ─────────────────────────────────────────────
def build_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model, base_model


# ─────────────────────────────────────────────
# STEP 4: FINE-TUNING
# ─────────────────────────────────────────────
def fine_tune(model, base_model):
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model


# ─────────────────────────────────────────────
# STEP 5: CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks():
    return [
        ModelCheckpoint(MODEL_SAVE, monitor="val_auc", save_best_only=True,
                        mode="max", verbose=1),
        EarlyStopping(monitor="val_auc", patience=5, mode="max",
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-7, verbose=1)
    ]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading dataset...")
    paths, labels = load_cbis_ddsm(MASS_CSV, DICOM_CSV, IMAGE_ROOT)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )

    print(f"Split: Train={len(train_paths)} | Val={len(val_paths)} | Test={len(test_paths)}\n")

    train_gen = MammogramGenerator(train_paths, train_labels, BATCH_SIZE, augment=True)
    val_gen   = MammogramGenerator(val_paths,   val_labels,   BATCH_SIZE, augment=False)
    test_gen  = MammogramGenerator(test_paths,  test_labels,  BATCH_SIZE, augment=False)

    model, base_model = build_model()
    model.summary()

    print("\n--- Phase 1: Training classification head ---")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=get_callbacks())

    print("\n--- Phase 2: Fine-tuning ResNet50 ---")
    model = fine_tune(model, base_model)
    model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=get_callbacks())

    print("\n--- Final Evaluation ---")
    results = model.evaluate(test_gen, verbose=1)
    for name, value in zip(["Loss", "Accuracy", "AUC", "Precision", "Recall"], results):
        print(f"  {name}: {value:.4f}")

    print(f"\nModel saved to: {MODEL_SAVE}")