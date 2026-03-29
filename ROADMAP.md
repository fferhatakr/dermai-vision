# DermaScan AI — Technical Roadmap

This document tracks the strategic evolution of DermaScan AI.
The project focuses on **Medical AI, MLOps, and Scalable Backend Systems**.

---

## Phase I: Foundation & Research (Completed)

> Core architecture established, multi-modal system research developed.

| Milestone | Description | Status |
|-----------|-------------|--------|
| Baseline Vision | Linear MLP → Custom CNN → ResNet18 transfer learning experiments | Completed |
| Multimodal Fusion (NLP) | DistilBERT patient anamnesis + image late fusion | Completed |
| Explainable AI (XAI) | Grad-CAM heatmap integration | Completed |
| CI/CD Pipeline | Pytest + GitHub Actions automated test infrastructure | Completed |
| Backbone Scaling | MobileNetV3-Large, 960-dim embeddings, 94.75% Recall@5 | Completed |
| Clinical Champion (v4) | EfficientNet-B3, 300×300, Focal Loss | Completed |
| Meta-Learner (v5) | XGBoost Late Fusion, 5-Fold CV, OOF stacking, Safety Override | Completed |

---

## Phase II: Engineering & System Integration (In Progress)

> Transitioning from individual research to a professional-grade product.

### Git Flow & Branch Strategy
- **Deliverable:** `main / dev / feature/*` branch structure.
- **Status:** Completed

### PostgreSQL + JWT Authentication
- **Deliverable:** Patient history via PostgreSQL, JWT Auth, user dashboard.
- **Status:** Completed

### Containerization & Docker Compose
- **Goal:** One-command production deployment with database.
- **Deliverable:** - Docker, Docker Hub, Hugging Face Spaces deployment
  - `docker-compose.yml` with API + PostgreSQL + Streamlit
  - Environment variable management via `.env`
  - Health checks and restart policies
- **Status:** Completed

### YOLO — Lesion Localization
- **Deliverable:** YOLOv8 bounding box detection, classification pipeline integration, center crop fallback for clinical artifacts.
- **Status:** Completed

### Data Expansion (25K ISIC 2019)
- **Goal:** Use full ISIC 2019 dataset without discarding majority class samples.
- **Deliverable:**
  - 25K images organized into 8-class folders
  - YOLO bulk cropping on full dataset
  - Updated metadata CSV with lesion_id grouping
  - WeightedRandomSampler handles class imbalance (no manual data deletion)
- **Status:** Completed

### Honest Evaluation Infrastructure
- **Deliverable:**
  - StratifiedGroupKFold (lesion-based split, no data leakage)
  - Separated train/val torchmetrics (metric contamination fix)
  - Correct ImageFolder index mapping
  - Threshold-optimized evaluation (MEL recall ≥80% via threshold=0.11)
  - TTA evaluation (5-view voting)
- **Baseline (honest):** 67% accuracy, 80.4% MEL recall (TTA + threshold)
- **Status:** Completed

---

## Phase III: Model Performance & Calibration (Month 1)

> Maximize model quality with honest evaluation. Target: 70–78% balanced accuracy.

### Training Optimization
- **Goal:** Fine-tune training pipeline for maximum performance on 25K dataset.
- **Deliverable:**
  - Learning rate warmup + cosine annealing
  - Mixup / CutMix regularization
  - Gradient accumulation tuning
- **Expected impact:** +1–3% accuracy
- **Status:** Planned

### Albumentations Integration
- **Goal:** Replace torchvision transforms with dermatology-specific augmentations.
- **Deliverable:**
  - Elastic deformation, coarse dropout, CLAHE
  - Hair artifact simulation
  - Advanced color augmentation for skin tone variation
- **Expected impact:** +2–3% accuracy
- **Status:** Planned

### XGBoost Meta-Learner Retrain
- **Goal:** Retrain XGBoost on new CNN's OOF predictions.
- **Deliverable:** Updated `xgb_meta_learner.json` and `xgb_features.pkl` consistent with current CNN.
- **Status:** Completed

### Confidence Calibration
- **Goal:** Ensure model confidence scores reflect true probability.
- **Deliverable:**
  - Temperature Scaling on validation set
  - Reliability diagrams (calibration curves)
  - MC Dropout for uncertainty estimation
  - Low-confidence rejection system ("model is not sure — consult a specialist")
- **Impact:** Prevents overconfident wrong predictions.
- **Status:** Planned

---

## Phase IV: MLOps & Production Performance (Month 2)

> Professional infrastructure for a continuously improving system.

### ONNX Inference Optimization
- **Goal:** Accelerate inference and enable cross-platform deployment.
- **Deliverable:** EfficientNet-B3 ONNX export, PyTorch vs ONNX speed benchmarking.
- **Status:** Completed
- **Result:** EfficientNet-B3 exported, max diff: 9.5e-07, cross-platform ready


### Three-Tier Analysis System
- **Goal:** Provide flexible analysis modes balancing speed and clinical depth.
- **Deliverable:**
  - **Quick Scan (ONNX):** Image-only, no metadata, ~50ms. Binary risk output. For rapid triage.
  - **Standard Analysis (ONNX + XGBoost):** Image + clinical metadata fusion. 8-class probability distribution.
  - **Detailed Analysis (PyTorch + XGBoost + Grad-CAM):** Full pipeline with heatmap. For suspicious cases and clinical presentation.
  - Out-of-distribution detection: YOLO-based rejection + confidence threshold guard
  - Mode selection via Streamlit UI
- **Expected impact:** Production-ready deployment flexibility, clinically meaningful UX
- **Status:** Completed

### MLflow Experiment Tracking
- **Goal:** Automatically log every training experiment.
- **Deliverable:**
  - Accuracy, loss, hyperparameters tracked per run
  - Model artifact versioning
  - Experiment comparison dashboard
- **Status:** Completed

### Model Monitoring & Drift Detection
- **Goal:** Detect when model performance degrades in production.
- **Deliverable:**
  - Evidently AI integration for data drift detection
  - API latency and confidence score monitoring
  - Automated alerts when distribution shifts
- **Status:** Planned

### Feedback Loop & Active Learning
- **Goal:** Build a model improvement cycle from real-world usage.
- **Deliverable:**
  - "Incorrect prediction" reporting button in Streamlit UI
  - Feedback storage in PostgreSQL
  - Data validation pipeline (filter noisy/invalid submissions)
  - Periodic retraining trigger (manual or scheduled)
- **Status:** Planned

### CI/CD Pipeline (Production-Grade)
- **Goal:** Automate testing, validation, and deployment.
- **Deliverable:**
  - GitHub Actions: lint + pytest + model validation on PR
  - Automated Docker image build and push
  - Model performance regression check before merge
- **Status:** Planned

### Test Coverage Expansion
- **Goal:** Reach ≥80% test coverage on critical paths.
- **Deliverable:**
  - API endpoint tests (auth, analyze, patients)
  - Model inference tests (shape, dtype, output range)
  - Data pipeline tests (transforms, CSV parsing, index mapping)
- **Status:** Planned

---

## Phase V: Research & Documentation (Month 3)

> Benchmarking, advanced techniques, and professional output.

### Ensemble Learning
- **Goal:** Combine multiple architectures for best performance.
- **Deliverable:**
  - EfficientNet-B3 + ConvNeXt-Tiny soft voting
  - Performance comparison: single model vs ensemble
  - Retrain XGBoost meta-learner on ensemble outputs
- **Requirement:** Cloud GPU (Vast.ai / Lambda Labs, ~$10–20 total)
- **Status:** Planned

### Technical Blog Post
- **Goal:** Publish findings and methodology for professional visibility.
- **Deliverable:**
  - "Honest Evaluation in Dermatology AI" or similar topic
  - Published on Medium / personal site / dev.to
  - Covers: metric pitfalls, threshold optimization, clinical safety tradeoffs
- **Status:** Planned

### Final Benchmark Report
- **Goal:** Document all experiments and results in a single reference.
- **Deliverable:**
  - All model versions compared (v4 → v8)
  - Confusion matrices, threshold curves, calibration plots
  - Lessons learned and failure analysis
  - PDF report suitable for portfolio or academic submission
- **Status:** Planned

### Vision Transformer Exploration (Stretch)
- **Goal:** Benchmark ViT against CNN approaches.
- **Deliverable:**
  - EfficientNet-B3 vs ViT-Small comparative study
  - Attention visualization vs Grad-CAM comparison
- **Note:** Only if ensemble results justify further architecture exploration.
- **Status:** Future

---

## Timeline Overview

```
Month 1 — Model & Deploy
  Week 1:  25K YOLO crop + train + evaluate
  Week 2:  Docker Compose deployment & Containerization
  Week 3:  Albumentations & Confidence Calibration
  Week 4:  XGBoost retrain & ONNX Inference Optimization

Month 2 — MLOps & Monitoring
  Week 5:  MLflow experiment tracking
  Week 6:  Evidently drift detection + feedback loop
  Week 7:  CI/CD pipeline (production-grade)
  Week 8:  Test coverage expansion (≥80%)

Month 3 — Research & Output
  Week 9:  Ensemble experiment (cloud GPU)
  Week 10: Technical blog post
  Week 11: Final benchmark report
  Week 12: Vision Transformer Exploration
```

---

## Progress Overview

```
Phase I    [====================] 100% — Completed
Phase II   [====================] 100% — Completed
Phase III  [=                   ]   5% — Starting
Phase IV   [                    ]   0% — Planned
Phase V    [                    ]   0% — Planned
```

---

*Last updated:— For detailed version history, see [CHANGELOG.md](CHANGELOG.md).*
