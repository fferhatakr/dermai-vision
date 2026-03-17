# DermaScan AI — Technical Roadmap

This document tracks the strategic evolution of DermaScan AI.  
The project focuses on **Medical AI, MLOps, and Scalable Backend Systems**.

---

## Phase I: Foundation & Research (Completed)

> Core architecture established, multimodal system developed, deployed to production.

| Milestone | Description | Status |
|-----------|-------------|--------|
| Baseline Vision | Linear MLP -> Custom CNN -> ResNet18 transfer learning experiments | Completed |
| Multimodal Fusion (NLP) | DistilBERT patient anamnesis + image late fusion | Completed |
| Explainable AI (XAI) | Grad-CAM heatmap integration | Completed |
| CI/CD Pipeline | Pytest + GitHub Actions automated test infrastructure | Completed |
| Backbone Scaling | MobileNetV3-Large, 960-dim embeddings, 94.75% Recall@5 | Completed |
| Clinical Champion (v4) | EfficientNet-B3, 300x300, Focal Loss, ~80% Melanoma Recall | Completed |
| Meta-Learner (v5) | XGBoost Late Fusion, 5-Fold CV, OOF stacking, Safety Override | Completed |
| Containerization | Docker, Docker Hub, Hugging Face Spaces deployment | Completed |

---

## Phase II: Engineering Robustness (In Progress)

> Transitioning the project from individual research to a professional-grade product.

### Git Flow & Branch Strategy
- **Goal:** Adopt professional development practices and protect the main branch.
- **Deliverable:** Establish `main / dev / feature/*` branch structure.
- **Status:** Completed

### PostgreSQL + JWT Authentication
- **Goal:** Transform the system into a persistent, multi-user product.
- **Deliverable:**
  - Patient history and analysis results stored via PostgreSQL
  - Doctor / user management system with JWT Auth
  - User-specific dashboard
- **Status:** Completed
- **Skills acquired:** SQL, authentication, backend security

### ONNX Inference Optimization
- **Goal:** Accelerate model inference and enable cross-platform deployment.
- **Deliverable:**
  - EfficientNet-B3 export to ONNX format
  - PyTorch vs ONNX inference speed benchmarking
  - CPU deployment optimization
- **Status:** Planned
- **Skills acquired:** Model optimization, edge deployment fundamentals

### YOLO — Lesion Localization
- **Goal:** Extend the system from classification to localization — not just "what" but "where".
- **Deliverable:**
  - YOLOv8 lesion bounding box detection
  - Classification + detection pipeline integration
  - Lesion boundary visualization in the clinical UI
- **Status:** Planned
- **Skills acquired:** Object detection, spatial localization

---

## Phase III: MLOps & Monitoring (Planned)

> Keeping the model healthy and observable in production.

### MLflow — Experiment Tracking
- **Goal:** Log every training experiment automatically.
- **Deliverable:** Accuracy, loss, and hyperparameters tracked per run via MLflow
- **Skills acquired:** Experiment tracking, model registry

### Model Monitoring & Drift Detection
- **Goal:** Monitor model performance in real-world usage over time.
- **Deliverable:** Data drift detection via Evidently AI or Arize, API latency and confidence logging
- **Skills acquired:** Production monitoring, day-2 operations

### Feedback Loop
- **Goal:** Build a model improvement cycle from user feedback.
- **Deliverable:** Incorrect diagnosis reporting mechanism, feedback-driven data collection, model update pipeline
- **Skills acquired:** Active learning fundamentals

---

## Phase IV: Advanced Research (Future)

> Second project and research phase. To begin after Phase II is complete.

### Vision Transformer (ViT)
- EfficientNet-B3 vs ViT comparative benchmarking
- To be applied in the PathScan histopathology project
- Study of attention mechanisms in medical imaging

### TensorRT — GPU Optimization
- Next step after ONNX
- Target: 10x inference speedup on NVIDIA GPUs
- Production-grade deployment pipeline

### Ensemble Learning
- EfficientNet + ViT + ConvNeXt soft voting
- UI visualization of model disagreements

---

## Progress Overview

```
Phase I    [====================] 100% - Completed
Phase II   [============        ]  60% - In Progress
Phase III  [                    ]   0% - Planned
Phase IV   [                    ]   0% - Future
```

---

*Last updated: v6.0.0 — For detailed version history, see [CHANGELOG.md](CHANGELOG.md).*
