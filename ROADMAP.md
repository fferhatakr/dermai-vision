# DermaScan AI — Technical Roadmap v7.0

This document tracks the strategic evolution of DermaScan AI.
The project focuses on **Medical AI, MLOps, and Scalable Backend Systems**.

---

## Phase I: Foundation & Research (Completed)

> Core architecture established, multimodal system developed, deployed to production.

| Milestone | Description | Status |
|-----------|-------------|--------|
| Baseline Vision | Linear MLP → Custom CNN → ResNet18 transfer learning experiments | Completed |
| Multimodal Fusion (NLP) | DistilBERT patient anamnesis + image late fusion | Completed |
| Explainable AI (XAI) | Grad-CAM heatmap integration | Completed |
| CI/CD Pipeline | Pytest + GitHub Actions automated test infrastructure | Completed |
| Backbone Scaling | MobileNetV3-Large, 960-dim embeddings, 94.75% Recall@5 | Completed |
| Clinical Champion (v4) | EfficientNet-B3, 300×300, Focal Loss | Completed |
| Meta-Learner (v5) | XGBoost Late Fusion, 5-Fold CV, OOF stacking, Safety Override | Completed |
| Containerization | Docker, Docker Hub, Hugging Face Spaces deployment | Completed |

---

## Phase II: Engineering Robustness (In Progress)

> Transitioning the project from individual research to a professional-grade product.

### Git Flow & Branch Strategy
- **Goal:** Adopt professional development practices and protect the main branch.
- **Deliverable:** `main / dev / feature/*` branch structure.
- **Status:** Completed

### PostgreSQL + JWT Authentication
- **Goal:** Transform the system into a persistent, multi-user product.
- **Deliverable:** Patient history via PostgreSQL, JWT Auth, user dashboard.
- **Status:** Completed

### YOLO — Lesion Localization
- **Goal:** Extend from classification to localization — not just "what" but "where".
- **Deliverable:** YOLOv8 bounding box detection, classification pipeline integration.
- **Status:** Completed

### Honest Evaluation Infrastructure
- **Goal:** Eliminate metric inflation, establish trustworthy benchmarks.
- **Deliverable:**
  - StratifiedGroupKFold (lesion-based split, no data leakage)
  - Separated train/val metrics (torchmetrics bug fix)
  - Correct ImageFolder index mapping
  - Threshold-optimized evaluation (MEL recall ≥80% via threshold=0.10)
- **Status:** Completed
- **Baseline (honest):** 62.6% accuracy, 80.5% MEL recall (with threshold)

### ONNX Inference Optimization
- **Goal:** Accelerate inference and enable cross-platform deployment.
- **Deliverable:** EfficientNet-B3 ONNX export, PyTorch vs ONNX benchmarking.
- **Status:** Planned

---

## Phase III: Model Performance Push (Planned — Priority)

> Maximize classification accuracy with honest evaluation. Target: 78–82% balanced accuracy.

### 3.1 Data Expansion
- **Goal:** Increase dataset size from 13K to 40K+ images.
- **Deliverable:** Integrate ISIC 2019–2020 archives, clean labels, merge metadata.
- **Expected impact:** +5–10% accuracy
- **Status:** Planned

### 3.2 Advanced Augmentation
- **Goal:** Reduce overfitting with dermatology-specific augmentation.
- **Deliverable:**
  - Rotation 90°, perspective, GaussianBlur, hue jitter
  - Mixup / CutMix integration
- **Expected impact:** +2–3% accuracy
- **Status:** In Progress

### 3.3 Backbone Upgrade
- **Goal:** Replace EfficientNet-B3 with stronger architecture.
- **Deliverable:**
  - ConvNeXt-Tiny (first candidate)
  - EfficientNet-V2-S (second candidate)
  - Head-to-head comparison on same data/split
- **Expected impact:** +3–5% accuracy
- **Status:** Planned

### 3.4 Training Optimization
- **Goal:** Extract maximum performance from current setup.
- **Deliverable:**
  - Cosine annealing scheduler (replace ReduceLROnPlateau)
  - Gradient accumulation tuning
  - Test Time Augmentation (TTA) at evaluation
- **Expected impact:** +2–3% accuracy
- **Status:** Planned

### 3.5 Ensemble Learning
- **Goal:** Combine multiple models for best overall performance.
- **Deliverable:**
  - EfficientNet-B3 + ConvNeXt-Tiny soft voting
  - Retrain XGBoost meta-learner on new CNN outputs
- **Expected impact:** +3–5% accuracy
- **Requirement:** Cloud GPU recommended (Vast.ai / Lambda Labs)
- **Status:** Planned

---

## Phase IV: MLOps & Monitoring (Planned)

> Keeping the model healthy and observable in production.

### MLflow — Experiment Tracking
- **Goal:** Log every training experiment automatically.
- **Deliverable:** Accuracy, loss, hyperparameters tracked per run.

### Model Monitoring & Drift Detection
- **Goal:** Monitor model performance in real-world usage.
- **Deliverable:** Data drift detection via Evidently AI, API latency logging.

### Feedback Loop
- **Goal:** Build model improvement cycle from user feedback.
- **Deliverable:** Incorrect diagnosis reporting, feedback-driven data collection.

---

## Phase V: Advanced Research (Future)

> To begin after Phase III is complete and production model is stable.

### Vision Transformer (ViT)
- EfficientNet-B3 vs ViT comparative benchmarking
- Attention mechanism study in medical imaging

### TensorRT — GPU Optimization
- 10× inference speedup target on NVIDIA GPUs
- Production-grade deployment pipeline

---

## Progress Overview

```
Phase I    [====================] 100% — Completed
Phase II   [================    ]  80% — In Progress
Phase III  [===                 ]  15% — Starting
Phase IV   [                    ]   0% — Planned
Phase V    [                    ]   0% — Future
```

---

*Last updated: v7.0.0 — For detailed version history, see [CHANGELOG.md](CHANGELOG.md).*
