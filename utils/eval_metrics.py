import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix

# ---- CONFIG

PRED_DIR = r"C:\Users\asage\Documents\.ABM\img_res"
GT_DIR   = r"C:\Users\asage\Documents\.ABM\nnUNet\nnUNet_raw\Dataset501_ARCADE\labelsTs"
NUM_CLASSES = 25
LABEL_OFFSET = 0

def load_mask(path):
    return np.array(Image.open(path))


# ---- METRICS

def compute_metrics(conf_matrix):
    eps = 1e-7

    TP = np.diag(conf_matrix)
    FP = conf_matrix.sum(axis=0) - TP
    FN = conf_matrix.sum(axis=1) - TP

    support = conf_matrix.sum(axis=1)  # GT frequency per class

    iou = TP / (TP + FP + FN + eps)
    dice = (2 * TP) / (2 * TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    accuracy = TP.sum() / (conf_matrix.sum() + eps)

    # ---- WEIGHTED METRICS

    weights = support / (support.sum() + eps)

    weighted_iou = np.sum(iou * weights)
    weighted_dice = np.sum(dice * weights)
    weighted_precision = np.sum(precision * weights)
    weighted_recall = np.sum(recall * weights)
    weighted_f1 = np.sum(f1 * weights)

    return {
        "accuracy": accuracy,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "mean_iou": np.mean(iou),
        "mean_dice": np.mean(dice),
        "mean_f1": np.mean(f1),
        "weighted_iou": weighted_iou,
        "weighted_dice": weighted_dice,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def evaluate():
    pred_files = sorted(os.listdir(PRED_DIR))
    gt_files = sorted(os.listdir(GT_DIR))

    assert len(pred_files) == len(gt_files), "Mismatch in number of files"

    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for p_file, g_file in zip(pred_files, gt_files):
        pred = load_mask(os.path.join(PRED_DIR, p_file)) - LABEL_OFFSET
        gt   = load_mask(os.path.join(GT_DIR, g_file)) - LABEL_OFFSET

        pred = pred.flatten()
        gt = gt.flatten()

        mask = (gt >= 0) & (gt < NUM_CLASSES)
        pred = pred[mask]
        gt = gt[mask]

        conf_matrix += confusion_matrix(
            gt, pred, labels=list(range(NUM_CLASSES))
        )

    m = compute_metrics(conf_matrix)

    print("\n===== GLOBAL METRICS =====")
    print(f"Pixel Accuracy   : {m['accuracy']:.4f}")
    print(f"Mean IoU         : {m['mean_iou']:.4f}")
    print(f"Mean Dice        : {m['mean_dice']:.4f}")
    print(f"Mean F1          : {m['mean_f1']:.4f}")

    print("\n===== WEIGHTED METRICS =====")
    print(f"Weighted IoU     : {m['weighted_iou']:.4f}")
    print(f"Weighted Dice    : {m['weighted_dice']:.4f}")
    print(f"Weighted Prec    : {m['weighted_precision']:.4f}")
    print(f"Weighted Recall  : {m['weighted_recall']:.4f}")
    print(f"Weighted F1      : {m['weighted_f1']:.4f}")

    df = pd.DataFrame({
        "Class": np.arange(NUM_CLASSES),
        "Support": m["support"],
        "IoU": m["iou"],
        "Dice": m["dice"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m["f1"],
    })

    print("\n===== PER-CLASS TABLE =====")
    print(df.to_string(index=False))

    # optional save
    df.to_csv("segmentation_metrics_report.csv", index=False)
    print("\nSaved: segmentation_metrics_report.csv")


if __name__ == "__main__":
    evaluate()  