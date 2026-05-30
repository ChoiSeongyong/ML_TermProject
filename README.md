# ML_TermProject

머신러닝 텀 프로젝트 — 카사바 잎 병해 5-class 분류

Kaggle [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) 데이터로 Majority baseline, ResNet-50, Swin-Tiny를 비교합니다. 
ImageNet 사전학습 가중치(`timm`)를 불러와 5-class로 fine-tuning했습니다.

실험 결과와 분석은 PDF 보고서에 정리했습니다.

## 실행

### 1. 환경 설치
필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

*주요 의존성: `torch`, `timm`, `albumentations`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `opencv-python`*

### 2. 데이터 준비
본 프로젝트는 Kaggle Cassava Leaf Disease Classification 데이터셋을 사용합니다. 데이터는 아래 두 가지 방법으로 준비할 수 있습니다.

* **방법 A (수동):** Kaggle에서 데이터를 다운로드하여 프로젝트 루트의 `data/` 폴더 아래에 `train.csv`와 `train_images/`를 직접 배치합니다.
```text
data/
  train.csv
  train_images/
```
* **방법 B (자동 API):** 본인의 Kaggle 계정에서 발급받은 `kaggle.json` 키 파일을 프로젝트 루트나 `~/.kaggle/`에 두면, 스크립트 실행 시 1번 셀에서 API를 통해 자동으로 다운로드 및 압축 해제를 진행합니다.

### 3. 실험 실행
전체 학습 및 평가 파이프라인을 실행합니다. (GPU 환경 권장)
python ml_term_project.py

---

## 파이프라인 요약

**데이터 파이프라인:**
   
   **Data Split:** Train / Val / Test = 8 : 1 : 1 비율로 계층적 분할 (Stratified Split)을 적용하여 클래스 비율 유지
   
   **Augmentation (Albumentations):** `RandomResizedCrop`, `HorizontalFlip`, `VerticalFlip`, `ShiftScaleRotate`, `Normalize`를 통한 강건한 전처리 및 증강 파이프라인 구축

**클래스 불균형 해결:** 각 클래스의 빈도 역수(Inverse-Frequency)를 기반으로 계산된 `weighted CrossEntropyLoss`를 적용하여 소수 클래스에 대한 학습력 강화 
**하이퍼파라미터 튜닝:** ResNet-50과 Swin-T 각각에 대해 **Validation Macro F1-Score**를 기준으로 Learning Rate Grid Search를 수행 (5 Epochs)한 뒤, 최적의 LR로 본 학습(20 Epochs) 진행
**다각적 성능 평가:** Accuracy, Macro/Weighted F1-Score, ROC-AUC (One-vs-Rest) 및 오분류 쌍 히트맵(Misclassification Pairs Heatmap) 분석

학습·평가 figure와 `test_metrics_summary.csv`, `experiment_config.json` 등은 실행 후 `figures/`에 저장됩니다.
