#  1. 환경 설정 및 데이터 준비
# !pip install -q kaggle albumentations timm scikit-learn seaborn

import os
import shutil
import zipfile

DATA_DIR = "./data"
IMG_DIR = os.path.join(DATA_DIR, "train_images")

FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def data_ready():
    csv_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(csv_path):
        return False
    if not os.path.isdir(IMG_DIR):
        return False
    n_imgs = sum(
        1 for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    return n_imgs >= 100


def setup_kaggle_credentials():
    """Colab 업로드, 프로젝트 루트, 또는 ~/.kaggle/ 에서 kaggle.json 탐색."""
    home_cfg = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists("kaggle.json"):
        src = "kaggle.json"
    elif os.path.exists(home_cfg):
        src = home_cfg
    else:
        try:
            from google.colab import files

            print("Kaggle API: 본인 계정에서 발급한 kaggle.json 을 업로드하세요.")
            uploaded = files.upload()
            if "kaggle.json" not in uploaded:
                raise FileNotFoundError("kaggle.json 업로드가 필요합니다.")
            src = "kaggle.json"
        except ImportError:
            raise FileNotFoundError(
                'kaggle.json 이 없습니다. README "데이터 준비"를 참고해 '
                "./data/train.csv 와 ./data/train_images/ 를 직접 배치하세요."
            )
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    shutil.copy(src, os.path.expanduser("~/.kaggle/kaggle.json"))
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)


if data_ready():
    n_imgs = len(
        [
            f
            for f in os.listdir(IMG_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    print(f"데이터 준비 완료 — train.csv + train_images/ ({n_imgs:,} images)")
else:
    print("=" * 64)
    print("데이터가 ./data/ 에 없습니다. 아래 중 하나를 선택하세요.")
    print("  [권장] 수동: Kaggle 대회 Data 탭 → train.csv, train_images.zip")
    print("         → ./data/ 에 배치 후 이 셀을 다시 실행")
    print("  [선택] API: 본인 kaggle.json 으로 자동 다운로드 (USE_KAGGLE_API=True)")
    print("  대회: https://www.kaggle.com/c/cassava-leaf-disease-classification")
    print("=" * 64)

    USE_KAGGLE_API = True
    if USE_KAGGLE_API:
        setup_kaggle_credentials()
        os.system(
            "kaggle competitions download -c cassava-leaf-disease-classification -p ."
        )
        zip_path = "cassava-leaf-disease-classification.zip"
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DATA_DIR)
            print("압축 해제 완료:", DATA_DIR)
        else:
            raise FileNotFoundError("Kaggle 다운로드 zip 파일을 찾을 수 없습니다.")
    else:
        raise FileNotFoundError(
            "USE_KAGGLE_API=False 인 경우 README대로 data/ 를 채운 뒤 다시 실행하세요."
        )

    if not data_ready():
        raise RuntimeError(
            "다운로드 후에도 data/train.csv 또는 train_images/ 가 비어 있습니다."
        )
    print("데이터 다운로드·검증 완료!")

#  2. 설정, 임포트, 재현성
import json
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import timm

plt.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid")

# ========== 하이퍼파라미터 ==========
SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
NUM_CLASSES = 5
IMG_SIZE = 224

# Learning rate: grid search 후 덮어씀 (기본값은 fallback)
LR_RESNET = 1e-4
LR_SWIN = 1e-4
WEIGHT_DECAY_SWIN = 1e-5

# ResNet LR 튜닝 (셀 6, 학습 전)
RUN_HP_TUNING = True
HP_TUNING_EPOCHS = 5
LR_CANDIDATES = [1e-5, 1e-4, 1e-3]

# Swin LR 튜닝 (셀 9, ResNet 학습 후) — ViT는 보통 CNN보다 낮은 LR
RUN_SWIN_HP_TUNING = True
SWIN_LR_CANDIDATES = [5e-5, 1e-4, 5e-4]

# 클래스 불균형: inverse-frequency 가중 CrossEntropy
USE_CLASS_WEIGHTS = True

LABEL_MAP = {
    0: "CBB (세균성 마름병)",
    1: "CBSD (갈색 줄무늬병)",
    2: "CGM (녹색 거미 진드기)",
    3: "CMD (모자이크병)",
    4: "Healthy (정상)",
}
LABEL_NAMES = [LABEL_MAP[i] for i in range(NUM_CLASSES)]


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")

#  3. 데이터 로드 · 분할 · EDA 시각화
df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
assert {"image_id", "label"}.issubset(
    df.columns
), "train.csv 에 image_id, label 컬럼이 필요합니다."

missing = [
    i for i in df["image_id"].head(500) if not os.path.exists(os.path.join(IMG_DIR, i))
]
if missing:
    raise FileNotFoundError(
        f"이미지 경로 불일치: 예) {missing[0]} — train_images/ 위치를 확인하세요."
    )
print(f"전체 샘플 수: {len(df):,}")
print(df.head())

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# --- Fig 1: 클래스 분포 (전체 / split별) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, sub) in zip(
    axes, [("Train", train_df), ("Val", val_df), ("Test", test_df)]
):
    counts = sub["label"].value_counts().sort_index()
    ax.bar(
        range(NUM_CLASSES),
        [counts.get(i, 0) for i in range(NUM_CLASSES)],
        color="steelblue",
    )
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"{i}" for i in range(NUM_CLASSES)], rotation=0)
    ax.set_title(f"{name} class distribution")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Count")
plt.suptitle("Dataset: Class Distribution by Split", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_distribution_splits.png", bbox_inches="tight")
plt.show()

# --- Fig 2: 전체 클래스 분포 (라벨명) ---
fig, ax = plt.subplots(figsize=(9, 4))
full_counts = df["label"].value_counts().sort_index()
bars = ax.barh(
    LABEL_NAMES,
    [full_counts[i] for i in range(NUM_CLASSES)],
    color=sns.color_palette("Set2", 5),
)
for b, c in zip(bars, full_counts):
    ax.text(c + 50, b.get_y() + b.get_height() / 2, str(c), va="center")
ax.set_title("Overall Class Distribution (Imbalance Check)")
ax.set_xlabel("Number of images")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_class_distribution_labeled.png", bbox_inches="tight")
plt.show()

# --- Fig 3: 클래스별 샘플 이미지 (5x3 grid) ---
fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(9, 14))
for cls in range(NUM_CLASSES):
    samples = df[df["label"] == cls].sample(3, random_state=SEED)["image_id"].tolist()
    for col, img_id in enumerate(samples):
        path = os.path.join(IMG_DIR, img_id)
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        axes[cls, col].imshow(img)
        axes[cls, col].axis("off")
        if col == 0:
            axes[cls, col].set_ylabel(LABEL_MAP[cls], fontsize=9)
plt.suptitle("Sample Images per Class (3 random per class)", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_sample_images_per_class.png", bbox_inches="tight")
plt.show()

# --- Fig 4: 이미지 해상도 분포 ---
sample_for_size = df.sample(min(800, len(df)), random_state=SEED)
widths, heights = [], []
for img_id in tqdm(sample_for_size["image_id"], desc="Sampling resolutions"):
    img = cv2.imread(os.path.join(IMG_DIR, img_id))
    if img is not None:
        h, w = img.shape[:2]
        heights.append(h)
        widths.append(w)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(widths, bins=30, color="coral", edgecolor="white")
axes[0].set_title("Image Width Distribution")
axes[1].hist(heights, bins=30, color="seagreen", edgecolor="white")
axes[1].set_title("Image Height Distribution")
plt.suptitle(f"Image Resolution Stats (n={len(widths)} samples)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_image_resolution_hist.png", bbox_inches="tight")
plt.show()
print(f"Median size: {int(np.median(widths))} x {int(np.median(heights))}")
#  4. Dataset · DataLoader · Augmentation
train_transforms = A.Compose(
    [
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

valid_transforms = A.Compose(
    [
        A.Resize(256, 256),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


class CassavaDataset(Dataset):
    def __init__(self, frame, data_dir, transforms=None):
        self.frame = frame.reset_index(drop=True)
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = os.path.join(self.data_dir, row["image_id"])
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        label = int(row["label"])
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, label

    def get_raw_image(self, index):
        """시각화용 (증강 없음)"""
        row = self.frame.iloc[index]
        path = os.path.join(self.data_dir, row["image_id"])
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), int(row["label"])


def make_loader(frame, transforms, shuffle):
    ds = CassavaDataset(frame, IMG_DIR, transforms)
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2, pin_memory=True
    )


train_loader = make_loader(train_df, train_transforms, shuffle=True)
val_loader = make_loader(val_df, valid_transforms, shuffle=False)
test_loader = make_loader(test_df, valid_transforms, shuffle=False)
test_ds = CassavaDataset(test_df, IMG_DIR, valid_transforms)
print("DataLoaders ready.")


#  5. 모델 정의 · 학습/평가 유틸
def make_class_weights(frame):
    counts = frame["label"].value_counts().sort_index()
    counts = counts.reindex(range(NUM_CLASSES), fill_value=1)
    weights = len(frame) / (NUM_CLASSES * counts.astype(float))
    return torch.tensor(weights.values, dtype=torch.float32)


CLASS_WEIGHTS = make_class_weights(train_df) if USE_CLASS_WEIGHTS else None
criterion = nn.CrossEntropyLoss(
    weight=CLASS_WEIGHTS.to(device) if CLASS_WEIGHTS is not None else None
)
if CLASS_WEIGHTS is not None:
    print(
        "Class weights (inverse freq):", [round(w, 3) for w in CLASS_WEIGHTS.tolist()]
    )


class BaselineResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class SwinModel(nn.Module):
    def __init__(
        self, model_name="swin_tiny_patch4_window7_224", num_classes=NUM_CLASSES
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


def run_epoch(model, loader, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train(is_train)
    losses, preds, targets = [], [], []

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        if is_train:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)

        losses.append(loss.item())
        preds.extend(logits.argmax(1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    if is_train and scheduler is not None:
        scheduler.step()

    return {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro"),
        "f1_weighted": f1_score(targets, preds, average="weighted"),
        "preds": np.array(preds),
        "targets": np.array(targets),
    }


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    probs, targets = [], []
    for images, labels in tqdm(loader, desc="Predict proba", leave=False):
        images = images.to(device)
        logits = model(images)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.extend(labels.numpy())
    return np.vstack(probs), np.array(targets)


def compute_metrics(targets, preds, probas=None):
    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro"),
        "f1_weighted": f1_score(targets, preds, average="weighted"),
    }
    if probas is not None:
        y_bin = label_binarize(targets, classes=list(range(NUM_CLASSES)))
        metrics["roc_auc_ovr"] = roc_auc_score(
            y_bin, probas, average="macro", multi_class="ovr"
        )
    return metrics


def train_model(model, name, optimizer, scheduler=None, epochs=EPOCHS, ckpt_path=None):
    history = defaultdict(list)
    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, scheduler)
        va = run_epoch(model, val_loader)

        for split, res in [("train", tr), ("val", va)]:
            history[f"{split}_loss"].append(res["loss"])
            history[f"{split}_acc"].append(res["acc"])
            history[f"{split}_f1"].append(res["f1_macro"])

        print(
            f"[{name}] Epoch {epoch:02d}/{epochs} | "
            f"Train Loss {tr['loss']:.4f} Acc {tr['acc']:.4f} F1 {tr['f1_macro']:.4f} | "
            f"Val Loss {va['loss']:.4f} Acc {va['acc']:.4f} F1 {va['f1_macro']:.4f}"
        )

        if va["f1_macro"] > best_val_f1 and ckpt_path:
            best_val_f1 = va["f1_macro"]
            torch.save(model.state_dict(), ckpt_path)

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded best checkpoint (Val Macro F1={best_val_f1:.4f})")

    history["best_val_f1_macro"] = best_val_f1
    return history


def tune_learning_rate(
    candidates,
    build_model,
    build_optimizer,
    model_label="Model",
    epochs=HP_TUNING_EPOCHS,
    save_csv="hyperparameter_lr_search.csv",
    save_fig=None,
):
    """Validation Macro F1 기준 learning rate 선택."""
    records = []
    best_lr, best_f1 = candidates[0], -1.0

    for lr in candidates:
        print(f"\n--- [{model_label}] HP tuning: lr={lr:g}, {epochs} epochs ---")
        model = build_model()
        optimizer = build_optimizer(model, lr)
        history = train_model(
            model, f"{model_label} LR={lr:g}", optimizer, epochs=epochs, ckpt_path=None
        )
        peak_f1 = max(history["val_f1"]) if history["val_f1"] else -1.0
        records.append(
            {
                "model": model_label,
                "learning_rate": lr,
                "epochs": epochs,
                "best_val_f1_macro": peak_f1,
                "final_val_f1_macro": (
                    history["val_f1"][-1] if history["val_f1"] else np.nan
                ),
            }
        )
        if peak_f1 > best_f1:
            best_f1, best_lr = peak_f1, lr
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    hp_df = pd.DataFrame(records)
    hp_df.to_csv(f"{FIG_DIR}/{save_csv}", index=False)
    print(f"\n[{model_label}] 선택된 LR: {best_lr:g} (best val Macro F1={best_f1:.4f})")

    if save_fig:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(
            hp_df["learning_rate"].astype(str),
            hp_df["best_val_f1_macro"],
            color="steelblue",
        )
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Best Val Macro F1")
        ax.set_title(f"{model_label} LR Grid Search ({epochs} epochs)")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/{save_fig}", bbox_inches="tight")
        plt.show()

    return best_lr, hp_df


def plot_training_history(history, title, save_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    specs = [
        ("loss", "Loss"),
        ("acc", "Accuracy"),
        ("f1", "Macro F1"),
    ]
    for ax, (key, ylab) in zip(axes, specs):
        ax.plot(epochs_range, history[f"train_{key}"], label="Train")
        ax.plot(epochs_range, history[f"val_{key}"], label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{save_name}", bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(targets, preds, title, save_name):
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=range(NUM_CLASSES),
        yticklabels=range(NUM_CLASSES),
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{save_name}", bbox_inches="tight")
    plt.show()
    return cm


def plot_roc_curves(probas, targets, title, save_name):
    y_bin = label_binarize(targets, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(NUM_CLASSES):
        try:
            auc_i = roc_auc_score(y_bin[:, i], probas[:, i])
            from sklearn.metrics import RocCurveDisplay

            RocCurveDisplay.from_predictions(
                y_bin[:, i], probas[:, i], ax=ax, name=f"Class {i} (AUC={auc_i:.3f})"
            )
        except ValueError:
            pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{save_name}", bbox_inches="tight")
    plt.show()


#  6. ResNet-50 하이퍼파라미터 튜닝 (Learning Rate Grid Search)
hp_results_resnet = None

if RUN_HP_TUNING:
    LR_RESNET, hp_results_resnet = tune_learning_rate(
        LR_CANDIDATES,
        build_model=lambda: BaselineResNet().to(device),
        build_optimizer=lambda m, lr: torch.optim.Adam(m.parameters(), lr=lr),
        model_label="ResNet-50",
        epochs=HP_TUNING_EPOCHS,
        save_csv="hyperparameter_lr_search_resnet.csv",
        save_fig="05_hyperparameter_lr_search_resnet.png",
    )
    print(hp_results_resnet.round(4))
else:
    print("RUN_HP_TUNING=False — LR_RESNET 기본값 사용")

print(f"ResNet LR: {LR_RESNET:g}")

#  7. 간단 베이스라인: Majority Class
majority_class = train_df["label"].mode()[0]
maj_preds = np.full(len(test_df), majority_class)
maj_targets = test_df["label"].values
maj_metrics = compute_metrics(maj_targets, maj_preds)
print("=== Majority Class Baseline (Test) ===")
for k, v in maj_metrics.items():
    print(f"  {k}: {v:.4f}")
print(f"  (always predicts class {majority_class}: {LABEL_MAP[majority_class]})")

#  8. ResNet-50 학습 (전이학습 fine-tuning)
resnet = BaselineResNet().to(device)
opt_resnet = torch.optim.Adam(resnet.parameters(), lr=LR_RESNET)

print(f"=== ResNet-50 Training (lr={LR_RESNET:g}, epochs={EPOCHS}) ===")
hist_resnet = train_model(
    resnet, "ResNet-50", opt_resnet, epochs=EPOCHS, ckpt_path="best_resnet.pth"
)
plot_training_history(
    hist_resnet, "ResNet-50 Learning Curves", "06_resnet_learning_curves.png"
)

#  9. Swin-T 하이퍼파라미터 튜닝 (Learning Rate Grid Search)
hp_results_swin = None

if RUN_SWIN_HP_TUNING:
    LR_SWIN, hp_results_swin = tune_learning_rate(
        SWIN_LR_CANDIDATES,
        build_model=lambda: SwinModel().to(device),
        build_optimizer=lambda m, lr: torch.optim.AdamW(
            m.parameters(), lr=lr, weight_decay=WEIGHT_DECAY_SWIN
        ),
        model_label="Swin-T",
        epochs=HP_TUNING_EPOCHS,
        save_csv="hyperparameter_lr_search_swin.csv",
        save_fig="07_hyperparameter_lr_search_swin.png",
    )
    print(hp_results_swin.round(4))
else:
    print("RUN_SWIN_HP_TUNING=False — LR_SWIN 기본값 사용")

experiment_config = {
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "img_size": IMG_SIZE,
    "lr_resnet": LR_RESNET,
    "lr_swin": LR_SWIN,
    "weight_decay_swin": WEIGHT_DECAY_SWIN,
    "use_class_weights": USE_CLASS_WEIGHTS,
    "run_resnet_hp_tuning": RUN_HP_TUNING,
    "run_swin_hp_tuning": RUN_SWIN_HP_TUNING,
    "hp_tuning_epochs": HP_TUNING_EPOCHS,
    "resnet_lr_candidates": LR_CANDIDATES if RUN_HP_TUNING else None,
    "swin_lr_candidates": SWIN_LR_CANDIDATES if RUN_SWIN_HP_TUNING else None,
    "split": "train/val/test = 8:1:1 stratified",
}
with open(f"{FIG_DIR}/experiment_config.json", "w", encoding="utf-8") as f:
    json.dump(experiment_config, f, indent=2, ensure_ascii=False)
print("Saved:", f"{FIG_DIR}/experiment_config.json")
print(f"Final LR — ResNet: {LR_RESNET:g}, Swin: {LR_SWIN:g}")

#  10. Swin Transformer 학습 (전이학습 fine-tuning)
swin = SwinModel().to(device)
opt_swin = torch.optim.AdamW(
    swin.parameters(), lr=LR_SWIN, weight_decay=WEIGHT_DECAY_SWIN
)
sched_swin = torch.optim.lr_scheduler.CosineAnnealingLR(opt_swin, T_max=EPOCHS)

print(f"=== Swin-T Training (lr={LR_SWIN:g}, epochs={EPOCHS}) ===")
hist_swin = train_model(
    swin,
    "Swin-T",
    opt_swin,
    scheduler=sched_swin,
    epochs=EPOCHS,
    ckpt_path="best_swin.pth",
)
plot_training_history(
    hist_swin, "Swin-T Learning Curves", "08_swin_learning_curves.png"
)

#  11. 테스트 평가 · 모델 비교 · 성능 시각화
results = {}
all_preds = {}
all_probas = {}
test_targets = test_df["label"].values

results["Majority"] = maj_metrics
all_preds["Majority"] = maj_preds

for name, model in [("ResNet-50", resnet), ("Swin-T", swin)]:
    model.eval()
    te = run_epoch(model, test_loader)
    probas, _ = predict_proba(model, test_loader)
    metrics = compute_metrics(te["targets"], te["preds"], probas)
    results[name] = metrics
    all_preds[name] = te["preds"]
    all_probas[name] = probas
    print(f"\n=== {name} (Test) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(
        classification_report(
            te["targets"], te["preds"], target_names=LABEL_NAMES, digits=4
        )
    )

# --- 결과 표 ---
results_df = pd.DataFrame(results).T
print(results_df.round(4))
results_df.round(4).to_csv(f"{FIG_DIR}/test_metrics_summary.csv")

summary = {
    "experiment_config": experiment_config,
    "test_metrics": {
        k: {m: float(v) for m, v in row.items()} for k, row in results.items()
    },
    "majority_class": int(majority_class),
}
with open(f"{FIG_DIR}/experiment_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("Saved:", f"{FIG_DIR}/experiment_summary.json")

# --- Fig 8: 모델별 지표 막대 그래프 ---
plot_df = results_df[["accuracy", "f1_macro", "f1_weighted"]].drop(
    index="Majority", errors="ignore"
)
if "roc_auc_ovr" in results_df.columns:
    plot_df = results_df[["accuracy", "f1_macro", "f1_weighted", "roc_auc_ovr"]].drop(
        index="Majority", errors="ignore"
    )

fig, ax = plt.subplots(figsize=(9, 5))
plot_df.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black", linewidth=0.5)
ax.set_ylim(0, 1.05)
ax.set_title("Model Comparison on Test Set")
ax.set_ylabel("Score")
ax.legend(loc="lower right")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_model_comparison_bar.png", bbox_inches="tight")
plt.show()

# --- Fig 8-9: Confusion Matrix ---
plot_confusion_matrix(
    test_targets,
    all_preds["ResNet-50"],
    "ResNet-50 Confusion Matrix (Test)",
    "10_resnet_confusion_matrix.png",
)
plot_confusion_matrix(
    test_targets,
    all_preds["Swin-T"],
    "Swin-T Confusion Matrix (Test)",
    "11_swin_confusion_matrix.png",
)

# --- ROC ---
plot_roc_curves(
    all_probas["ResNet-50"], test_targets, "ResNet-50 ROC (OvR)", "12_resnet_roc.png"
)
plot_roc_curves(
    all_probas["Swin-T"], test_targets, "Swin-T ROC (OvR)", "13_swin_roc.png"
)

# --- Fig 12: 클래스별 F1 (Swin) ---
from sklearn.metrics import precision_recall_fscore_support

p, r, f1, sup = precision_recall_fscore_support(
    test_targets, all_preds["Swin-T"], labels=list(range(NUM_CLASSES))
)
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(NUM_CLASSES)
ax.bar(x - 0.2, f1, width=0.35, label="Swin-T F1", color="teal")
_, _, f1_r, _ = precision_recall_fscore_support(
    test_targets, all_preds["ResNet-50"], labels=list(range(NUM_CLASSES))
)
ax.bar(x + 0.2, f1_r, width=0.35, label="ResNet-50 F1", color="orange")
ax.set_xticks(x)
ax.set_xticklabels([f"{i}" for i in range(NUM_CLASSES)])
ax.set_xlabel("Class")
ax.set_ylabel("F1")
ax.set_title("Per-Class F1: ResNet-50 vs Swin-T")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/14_per_class_f1_comparison.png", bbox_inches="tight")
plt.show()


#  12. 예측 시각화 · 실패 모드 분석
def collect_misclassified(model, frame, n_max=None):
    """test set에서 오분류 인덱스 수집"""
    ds = CassavaDataset(frame, IMG_DIR, valid_transforms)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model.eval()
    failures, correct = [], []
    offset = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            labels_np = labels.numpy()
            for j in range(len(labels_np)):
                idx = offset + j
                rec = (idx, labels_np[j], preds[j])
                if preds[j] != labels_np[j]:
                    failures.append(rec)
                else:
                    correct.append(rec)
            offset += len(labels_np)
    return failures, correct


def show_prediction_grid(
    records,
    frame,
    model_name,
    title_prefix,
    n_show=12,
    save_name=None,
    wrong_only=False,
):
    n_show = min(n_show, len(records))
    if n_show == 0:
        print("No samples to show.")
        return
    samples = random.sample(records, n_show)
    cols = 4
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    raw_ds = CassavaDataset(frame, IMG_DIR, transforms=None)
    for ax_idx, (df_idx, true_lbl, pred_lbl) in enumerate(samples):
        img, _ = raw_ds.get_raw_image(df_idx)
        ax = axes[ax_idx]
        ax.imshow(img)
        is_wrong = true_lbl != pred_lbl
        color = "red" if is_wrong else "green"
        ax.set_title(
            f"True: {LABEL_MAP[true_lbl]}\nPred: {LABEL_MAP[pred_lbl]}",
            fontsize=8,
            color=color,
        )
        ax.axis("off")
    for ax in axes[n_show:]:
        ax.axis("off")
    plt.suptitle(f"{title_prefix} — {model_name}", fontsize=13)
    plt.tight_layout()
    if save_name:
        plt.savefig(f"{FIG_DIR}/{save_name}", bbox_inches="tight")
    plt.show()


fail_swin, ok_swin = collect_misclassified(swin, test_df)
print(
    f"Swin-T — Test misclassified: {len(fail_swin)} / {len(test_df)} ({100*len(fail_swin)/len(test_df):.1f}%)"
)

# --- Fig 13: 정답 예측 12장 ---
show_prediction_grid(
    ok_swin,
    test_df,
    "Swin-T",
    "Correct Predictions",
    n_show=12,
    save_name="15_swin_correct_predictions.png",
)

# --- 오분류 샘플 ---
show_prediction_grid(
    fail_swin,
    test_df,
    "Swin-T",
    "Misclassified (Failure Modes)",
    n_show=12,
    save_name="16_swin_misclassified.png",
    wrong_only=True,
)

# --- Fig 15: ResNet vs Swin 같은 이미지 비교 (6장) ---
compare_idx = random.sample(range(len(test_df)), min(6, len(test_df)))
fig, axes = plt.subplots(6, 3, figsize=(11, 22))
raw_ds = CassavaDataset(test_df, IMG_DIR, transforms=None)
resnet.eval()
swin.eval()

for row, idx in enumerate(compare_idx):
    img_raw, true_lbl = raw_ds.get_raw_image(idx)
    img_t = valid_transforms(image=img_raw)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        pr = resnet(img_t).argmax(1).item()
        ps = swin(img_t).argmax(1).item()
    axes[row, 0].imshow(img_raw)
    axes[row, 0].set_title(f"True: {LABEL_MAP[true_lbl]}", fontsize=9)
    axes[row, 0].axis("off")
    for col, (pred, name) in enumerate([(pr, "ResNet-50"), (ps, "Swin-T")], start=1):
        axes[row, col].imshow(img_raw)
        c = "green" if pred == true_lbl else "red"
        axes[row, col].set_title(
            f"{name}\nPred: {LABEL_MAP[pred]}", fontsize=9, color=c
        )
        axes[row, col].axis("off")
if compare_idx:
    axes[0, 1].set_ylabel("Model predictions", fontsize=10)
plt.suptitle("Side-by-Side: ResNet-50 vs Swin-T on Test Samples", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/17_resnet_vs_swin_comparison.png", bbox_inches="tight")
plt.show()

# --- Fig 16: 오분류 클래스 쌍 히트맵 (Swin) ---
if fail_swin:
    pairs = [(t, p) for _, t, p in fail_swin]
    pair_df = (
        pd.DataFrame(pairs, columns=["true", "pred"])
        .value_counts()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pair_df, annot=True, fmt="d", cmap="Reds", ax=ax)
    ax.set_title("Swin-T: Misclassification Pairs (True → Pred)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/18_misclassification_pairs.png", bbox_inches="tight")
    plt.show()

print(f"\n모든 figure 저장 위치: {FIG_DIR}/")
print("metrics:", results_df.round(4).to_dict())
