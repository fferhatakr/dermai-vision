import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

PH2_ROOT  = "data/ph2_data/PH2Dataset"
XLSX_PATH = "data/ph2_data/PH2_dataset.xlsx"
ONNX_PATH = "models/onnx_model/midas_onnx"
FINETUNE_CKPT = "models/vision/DermScan_finetune_derm12345.ckpt"

COMPARE_MODE = True

ISIC_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

PH2_TO_ISIC = {
    0: "NV",
    1: "NV",
    2: "MEL",
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MEL_THRESHOLD = 0.25


def preprocess(image_path, size=300):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]


def find_bmp(image_id):
    derm_dir = os.path.join(PH2_ROOT, image_id, f"{image_id}_Dermoscopic_Image")
    bmp_path = os.path.join(derm_dir, f"{image_id}.bmp")
    if os.path.exists(bmp_path):
        return bmp_path
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(derm_dir, image_id + ext)
        if os.path.exists(p):
            return p
    return None


def load_ph2_labels():
    df = pd.read_excel(XLSX_PATH, header=None)
    data = df.iloc[13:].copy()
    data = data.reset_index(drop=True)

    labels = {}
    for _, row in data.iterrows():
        img_id = str(row[0]).strip()
        if img_id == "nan" or not img_id.startswith("IMD"):
            continue

        common   = str(row[2]).strip()
        atypical = str(row[3]).strip()
        melanoma = str(row[4]).strip()

        if melanoma == "X":
            labels[img_id] = 2
        elif atypical == "X":
            labels[img_id] = 1
        elif common == "X":
            labels[img_id] = 0

    from collections import Counter
    dist = Counter(labels.values())
    print(f"labels_loaded: {len(labels)}")
    print(f"nv_common: {dist.get(0, 0)}")
    print(f"nv_atypical: {dist.get(1, 0)}")
    print(f"melanoma: {dist.get(2, 0)}")

    return labels


def run_ph2_test():
    try:
        import onnxruntime as ort
    except ImportError:
        print("error: install onnxruntime")
        return

    print(f"\n{'='*60}")
    print("ph2 cross dataset test")
    print(f"{'='*60}\n")

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    labels = load_ph2_labels()
    if not labels:
        print("error: labels not loaded")
        return

    y_true, y_pred, y_pred_thresh = [], [], []
    not_found = 0

    for img_id, cls_code in tqdm(labels.items()):
        bmp = find_bmp(img_id)
        if bmp is None:
            not_found += 1
            continue

        try:
            inp    = preprocess(bmp)
            logits = sess.run([output_name], {input_name: inp})[0][0]
        except Exception:
            not_found += 1
            continue

        e_    = np.exp(logits - logits.max())
        probs = e_ / e_.sum()

        pred_idx   = int(np.argmax(probs))
        pred_label = ISIC_CLASSES[pred_idx]

        mel_idx  = ISIC_CLASSES.index("MEL")
        mel_prob = float(probs[mel_idx])
        pred_thresh = "MEL" if mel_prob >= MEL_THRESHOLD else pred_label

        true_label = PH2_TO_ISIC[cls_code]
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_pred_thresh.append(pred_thresh)

    if not y_true:
        print("error: no images processed")
        return

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score

    present = sorted(set(y_true))

    acc_std    = accuracy_score(y_true, y_pred)
    acc_thresh = accuracy_score(y_true, y_pred_thresh)

    mel_rec_std    = recall_score(y_true, y_pred, labels=["MEL"], average=None, zero_division=0)
    mel_rec_thresh = recall_score(y_true, y_pred_thresh, labels=["MEL"], average=None, zero_division=0)
    mel_f1_std     = f1_score(y_true, y_pred, labels=["MEL"], average=None, zero_division=0)
    mel_f1_thresh  = f1_score(y_true, y_pred_thresh, labels=["MEL"], average=None, zero_division=0)

    print(f"count: {len(y_true)}")
    print(f"accuracy_argmax: {acc_std:.4f}")
    print(f"accuracy_thresh: {acc_thresh:.4f}")
    print(f"mel_recall_argmax: {mel_rec_std[0]:.4f}")
    print(f"mel_recall_thresh: {mel_rec_thresh[0]:.4f}")
    print(f"mel_f1_argmax: {mel_f1_std[0]:.4f}")
    print(f"mel_f1_thresh: {mel_f1_thresh[0]:.4f}")

    print("\nclassification_report")
    print(classification_report(y_true, y_pred_thresh, labels=present, target_names=present, zero_division=0))

    cm    = confusion_matrix(y_true, y_pred_thresh, labels=present)
    cm_df = pd.DataFrame(cm, index=present, columns=present)
    print("\nconfusion_matrix")
    print(cm_df.to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=axes[0], linewidths=0.5)
        axes[0].set_title("confusion_matrix")
        axes[0].set_xlabel("pred")
        axes[0].set_ylabel("true")

        per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
        axes[1].barh(present, per_class_acc * 100)
        axes[1].set_xlabel("accuracy_percent")
        axes[1].set_title("per_class_accuracy")
        axes[1].set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig("ph2_results.png", dpi=150, bbox_inches="tight")
        print("plot_saved: ph2_results.png")
    except Exception as e:
        print(f"plot_error: {e}")

    summary = {
        "dataset": "PH2",
        "overlap": 0,
        "test_size": len(y_true),
        "not_found": not_found,
        "mel_threshold": MEL_THRESHOLD,
        "accuracy_argmax": round(acc_std, 4),
        "accuracy_threshold": round(acc_thresh, 4),
        "mel_recall_argmax": round(float(mel_rec_std[0]), 4),
        "mel_recall_threshold": round(float(mel_rec_thresh[0]), 4),
        "mel_f1_argmax": round(float(mel_f1_std[0]), 4),
        "mel_f1_threshold": round(float(mel_f1_thresh[0]), 4)
    }

    with open("ph2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("summary_saved: ph2_summary.json")


def run_pytorch_inference(image_path, model, device, size=300):
    import torch

    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)

    tensor = torch.tensor(arr).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor).cpu().numpy()[0]

    e_ = np.exp(logits - logits.max())
    return e_ / e_.sum()


def run_comparison():
    import torch
    import sys
    sys.path.append(os.getcwd())

    try:
        import onnxruntime as ort
        from engine.trainer_core import DermatologLightning
    except ImportError as e:
        print(f"error: {e}")
        return

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ft_model = DermatologLightning.load_from_checkpoint(
        FINETUNE_CKPT,
        map_location=device,
        strict=False
    )
    ft_model.eval()
    ft_model.to(device)

    labels = load_ph2_labels()
    if not labels:
        print("error: labels not loaded")
        return

    mel_idx = ISIC_CLASSES.index("MEL")

    y_true = []
    y_pred_onnx_argmax = []
    y_pred_onnx_thresh = []
    y_pred_ft_argmax = []
    y_pred_ft_thresh = []

    for img_id, cls_code in tqdm(labels.items()):
        bmp = find_bmp(img_id)
        if bmp is None:
            continue

        try:
            inp = preprocess(bmp)
            logits_onnx = sess.run([output_name], {input_name: inp})[0][0]
            e_ = np.exp(logits_onnx - logits_onnx.max())
            probs_onnx = e_ / e_.sum()

            probs_ft = run_pytorch_inference(bmp, ft_model, device)
        except Exception:
            continue

        true_label = PH2_TO_ISIC[cls_code]
        y_true.append(true_label)

        for probs, argmax_list, thresh_list in [
            (probs_onnx, y_pred_onnx_argmax, y_pred_onnx_thresh),
            (probs_ft, y_pred_ft_argmax, y_pred_ft_thresh),
        ]:
            pred_idx = int(np.argmax(probs))
            pred_argmax = ISIC_CLASSES[pred_idx]
            mel_prob = float(probs[mel_idx])
            pred_thresh = "MEL" if mel_prob >= MEL_THRESHOLD else pred_argmax

            argmax_list.append(pred_argmax)
            thresh_list.append(pred_thresh)

    from sklearn.metrics import accuracy_score, recall_score, f1_score

    results = {}
    for name, y_pred in [
        ("isic_argmax", y_pred_onnx_argmax),
        ("isic_thresh", y_pred_onnx_thresh),
        ("ft_argmax", y_pred_ft_argmax),
        ("ft_thresh", y_pred_ft_thresh),
    ]:
        acc = accuracy_score(y_true, y_pred)
        mel_rec = recall_score(y_true, y_pred, labels=["MEL"], average=None, zero_division=0)[0]
        mel_f1 = f1_score(y_true, y_pred, labels=["MEL"], average=None, zero_division=0)[0]
        results[name] = {"acc": acc, "mel_rec": mel_rec, "mel_f1": mel_f1}

    print("comparison_results")
    for k, v in results.items():
        print(k, v)

    comparison = {
        "test_size": len(y_true),
        "mel_threshold": MEL_THRESHOLD,
        "results": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in results.items()}
    }

    with open("ph2_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print("comparison_saved: ph2_comparison.json")


if __name__ == "__main__":
    if COMPARE_MODE:
        run_comparison()
    else:
        run_ph2_test()