import os, random, logging
import numpy as np
import pandas as pd

# ---------- TensorFlow / Keras import (mendukung TF2.15 atau Keras3) ----------
try:
    # Keras 3 (TF >= 2.17)
    import tensorflow as tf
    import keras
    from keras import layers, regularizers, callbacks, optimizers
    KERAS_FLAVOR = "keras3"
except Exception:
    # TF 2.15 dan sebelumnya
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks, optimizers 
    KERAS_FLAVOR = "tf.keras"

# Sembunyikan warning yang tidak penting
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel(logging.ERROR)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

CSV_PATH = "processed_kelulusan.csv"
df = pd.read_csv(CSV_PATH)
if "Lulus" not in df.columns:
    raise ValueError("Kolom target 'Lulus' tidak ditemukan di processed_kelulusan.csv")

X = df.drop("Lulus", axis=1).astype("float32")
y = df["Lulus"].astype("int32")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train).astype("float32")
X_val_s   = sc.transform(X_val).astype("float32")
X_test_s  = sc.transform(X_test).astype("float32")
y_train   = y_train.values.astype("int32")
y_val     = y_val.values.astype("int32")
y_test    = y_test.values.astype("int32")

input_dim = X_train_s.shape[1]
print(f"[INFO] Shapes train/val/test: {X_train_s.shape} {X_val_s.shape} {X_test_s.shape}")
print(f"[INFO] Using: {KERAS_FLAVOR}  |  TF={tf.__version__}  |  Keras={keras.__version__}")

# ---------- 2) Class weight (optional bila imbalanced) ----------
classes = np.unique(y_train)
cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
print("[INFO] class_weight:", class_weight)

# ---------- 3) Model builder ----------
def build_model(input_dim: int, neurons=32, dropout=0.3, l2_lambda=0.0, use_bn=False):
    reg = regularizers.l2(l2_lambda) if l2_lambda > 0 else None
    m = keras.Sequential(name="kelulusan_nn")
    m.add(layers.Input(shape=(input_dim,)))
    if use_bn: m.add(layers.BatchNormalization())
    m.add(layers.Dense(neurons, activation="relu", kernel_regularizer=reg))
    if use_bn: m.add(layers.BatchNormalization())
    m.add(layers.Dropout(dropout))
    m.add(layers.Dense(max(neurons // 2, 8), activation="relu", kernel_regularizer=reg))
    if use_bn: m.add(layers.BatchNormalization())
    m.add(layers.Dense(1, activation="sigmoid"))
    return m

def get_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    raise ValueError("Optimizer tidak dikenal. Gunakan 'adam' atau 'sgd'.")

# ---------- 4) Grid eksperimen ----------
neurons_list = [32, 64, 128]
opt_settings = [
    ("adam", 1e-3),
    ("adam", 3e-4),
    ("sgd", 1e-2),
    ("sgd", 5e-3),
]
regularizations = [
    (0.30, 0.0,  False),
    (0.50, 0.0,  False),
    (0.30, 1e-4, False),
    (0.30, 0.0,  True),
]

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

results = []
best = {"key": None, "model": None, "hist": None, "val_auc": -np.inf, "val_f1": -np.inf}

run_id = 0
for neu in neurons_list:
    for (opt_name, lr) in opt_settings:
        for (drop, l2l, bn) in regularizations:
            run_id += 1
            key = f"run{run_id:03d}_neu{neu}_opt{opt_name}_lr{lr}_do{drop}_l2{l2l}_bn{bn}"
            print(f"\n=== {key} ===")

            model = build_model(input_dim, neurons=neu, dropout=drop, l2_lambda=l2l, use_bn=bn)
            opt = get_optimizer(opt_name, lr)
            model.compile(optimizer=opt, loss="binary_crossentropy")

            hist = model.fit(
                X_train_s, y_train,
                validation_data=(X_val_s, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0,
                class_weight=class_weight
            )
            val_proba  = model.predict(X_val_s,  verbose=0).ravel()
            test_proba = model.predict(X_test_s, verbose=0).ravel()

            val_pred  = (val_proba  >= 0.5).astype(int)
            test_pred = (test_proba >= 0.5).astype(int)

            v_auc = roc_auc_score(y_val,  val_proba)
            t_auc = roc_auc_score(y_test, test_proba)
            v_f1  = f1_score(y_val,  val_pred)
            t_f1  = f1_score(y_test, test_pred)
            v_acc = accuracy_score(y_val,  val_pred)
            t_acc = accuracy_score(y_test, test_pred)

            results.append({
                "run": key, "neurons": neu, "optimizer": opt_name, "lr": lr,
                "dropout": drop, "l2": l2l, "batchnorm": bn,
                "val_acc": float(v_acc), "val_auc": float(v_auc), "val_f1": float(v_f1),
                "test_acc": float(t_acc), "test_auc": float(t_auc), "test_f1": float(t_f1),
                "best_epoch": int(np.argmin(hist.history["val_loss"]) + 1),
            })
            
            if (v_auc > best["val_auc"]) or (np.isclose(v_auc, best["val_auc"]) and v_f1 > best["val_f1"]):
                best = {"key": key, "model": model, "hist": hist, "val_auc": v_auc, "val_f1": v_f1}

# ---------- 5) Rekap hasil ----------
res_df = pd.DataFrame(results).sort_values(["val_auc", "val_f1"], ascending=False)
print("\nTop 10 (sort by Val AUC → Val F1):")
print(res_df.head(10)[[
    "run","neurons","optimizer","lr","dropout","l2","batchnorm",
    "val_auc","val_f1","test_auc","test_f1"
]])

res_df.to_csv("nn_results.csv", index=False)
print("[SAVE] nn_results.csv")

# ---------- 6) Evaluasi lengkap model terbaik ----------
print(f"\n[BEST] {best['key']}  (Val AUC={best['val_auc']:.4f}, Val F1={best['val_f1']:.4f})")
best_model = best["model"]

y_test_proba = best_model.predict(X_test_s, verbose=0).ravel()
y_test_pred  = (y_test_proba >= 0.5).astype(int)
print("\n=== TEST REPORT (Best Model) ===")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_test_proba))
print("F1       :", f1_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, digits=3))

# ---------- 7) Plot learning curve ----------
hist = best["hist"]
plt.figure()
plt.plot(hist.history["loss"],     label="Train Loss")
plt.plot(hist.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title(f"Learning Curve — {best['key']}")
plt.tight_layout()
plt.savefig("learning_curve_best.png", dpi=120)
print("[SAVE] learning_curve_best.png")

# ---------- 8) Simpan model & scaler ----------
best_model.save("best_nn_model.keras")
print("[SAVE] best_nn_model.keras")

try:
    import joblib
    joblib.dump(sc, "scaler.joblib")
    print("[SAVE] scaler.joblib")
except Exception:
    print("[WARN] joblib tidak tersedia; lewati penyimpanan scaler (pip install joblib untuk menyimpan).")
