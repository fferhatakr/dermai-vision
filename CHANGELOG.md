# Changelog

All notable architectural changes, pipeline upgrades, and model iterations for the Skin Cancer Detection project will be documented in this file. The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.
## [v7.1.0] - Dockerization, CI/CD Setup & Architecture Refactoring

### Added
- **Docker Integration:** Introduced `docker-compose.yml` for seamless multi-container deployment (FastAPI + PostgreSQL).
- **Automated Testing:** Added a comprehensive Pytest suite (`test_api.py`, `test_data.py`, `test_model.py`) for API and DB validation.
- **Configuration Management:** Centralized settings with the new `configs/` directory structure.
- **Meta-Learner Scripts:** Introduced `extract_features.py` and `train_meta_learner.py` for advanced ensemble strategies.

### Changed
- **API & Backend:** Refactored `database.py`, `models.py`, and `inference.py` to flawlessly communicate within the Docker network.
- **CI/CD Pipeline:** Updated `.github/workflows/python-app.yml` to support automated testing with a service database.
- **Training Engine:** Streamlined `full_train.py` and `full_evaluate.py` for the new data structures.
- **Documentation:** Updated `README.md` and `ROADMAP.md` to reflect Phase II progress.

### Removed (Clean-up)
- **Legacy Scripts:** Deleted obsolete files (`bulk_crop.py`, `create_meta_dataset.py`, `export_to_onnx.py`) to reduce technical debt.
- **Old Inference Modules:** Removed redundant hybrid prediction scripts (`inference/__init__.py`, `hybrid_predict.py`) in favor of the unified API structure.



## [v7.0.0] - Honest Evaluation & Threshold Optimization

### Bug Fixes
- **Metric contamination fixed:** Separated train/val torchmetrics instances — val_acc was inflated by train state leaking into validation.
- **Index mapping fixed:** ImageFolder indices now correctly mapped to metadata CSV after filtering missing files.
- **YOLO global scope fix:** `yolo_model` added to global declaration in `load_ai_models()`.

### Evaluation Infrastructure
- **StratifiedGroupKFold:** Replaced StratifiedKFold with lesion-based grouping to prevent data leakage.
- **Threshold optimization:** MEL threshold tuned to 0.11 for ≥80% recall (was hardcoded 0.45).
- **TTA evaluation:** 5-view Test-Time Augmentation added to evaluation (horizontal/vertical flip, 90°/270° rotation).
- **Unified evaluation script:** `evaluate_full.py` replaces `evaluate_kfold.py` and `evaluate_with_threshold.py`.

### Training
- **Backbone selection:** `trainer_core.py` now accepts `backbone` parameter ("efficientnet_b3" or "convnext_tiny").
- **Cosine annealing:** Added as scheduler option alongside ReduceLROnPlateau.
- **ConvNeXt-Tiny:** `DermaScanModelV4` added to `vision_models.py`.
- **Configurable training:** `train_kfold_v3.py` with BACKBONE, LR, EXPERIMENT_NAME variables.

### Production
- **MEL safety threshold:** `analyze.py` updated from `prob_mel > 0.45` to `prob_mel > 0.11`.
- **Model checkpoint:** Updated to v7 TTA-validated model.

### Metrics (Honest, Fold 1)
| Metric | Standard | TTA + Threshold |
|--------|----------|-----------------|
| Accuracy | 62.6% | 67% |
| MEL Recall | 44.9% | 80.4% |
| MEL F1 | 0.46 | 0.72 |

## [v6.0.0] - Full-Stack Clinical System

### Major Additions
- **PostgreSQL Database Integration:** Three core tables established: `users`, `patients`, `analyses`.
- **JWT Authentication:** Doctor registration, login, and Bearer token protection on all sensitive endpoints.
- **API Modular Refactor:** `main.py` split into `models.py`, `inference.py`, `gradcam.py`, `schemas.py`, and `routes/` directory.
- **Patient Management:** `POST /patients/add`, `GET /patients/list`, `GET /patients/{id}/analyses` endpoints.
- **Smart Patient Flow:** Analyze endpoint auto-creates or reuses patient records by name.
- **Secure Analysis Endpoint:** `POST /analyze` requires JWT token. Results saved to database with doctor and patient relationship.
- **Git Flow:** `main / dev / feature/*` branch structure established.

### UI
- Login and register tabs added to Streamlit.
- Patient full name input added to analysis form.
- Automatic session expiry handling.


## [v5.1.0] - Containerization & Production Readiness
### Added
- **Dockerized Backend (FastAPI):** Created a multi-stage `python:3.10-slim` Dockerfile to encapsulate the inference engine, including PyTorch and XGBoost dependencies.
- **Environment Standardization:** Locked all dependencies in `requirements.txt` to ensure 100% reproducibility across OS environments (Windows/Linux/macOS).
- **Docker Hub Integration:** Successfully pushed the core API image (`technull1/dermascan-api`) to Docker Hub for seamless cloud deployment.
- **Security & Privacy (Shielding):** Implemented strict `.dockerignore` and `.gitignore` policies to prevent leakage of sensitive environment variables (`.env`) and local training artifacts.
- **Automated Test Samples:** Integrated an internal sample selector in the UI to allow instant testing of high-risk cases without requiring manual user uploads.

### Improved
- **Cold Boot Optimization:** Leveraged Docker layer caching to reduce build times for iterative updates.
- **Image Optimization:** Switched to a slim Linux-based container, significantly reducing the production footprint.

## [v5.0.0] - The "Meta-Learner & Clinical Safety" Update (Current)
###  Major Pivot
- **Multimodal Late Fusion:** Deprecated the free-text NLP/DistilBERT symptom analysis. Transitioned to structured clinical metadata (Age, Sex, Anatomical Site) using an **XGBoost Meta-Learner** to evaluate probabilistic combinations.
- **Scientific Robustness (5-Fold CV):** Abandoned simple train/test splits in favor of a rigorous **5-Fold Cross-Validation** pipeline to scientifically validate the ~82% Melanoma Recall rate.
- **Data Leakage Prevention:** Trained the XGBoost meta-learner strictly on **Out-Of-Fold (OOF) predictions** generated by the CNN to eliminate stacking data leakage.

### Additions and Corrections
- **Safety Override Logic:** Implemented a hardcoded clinical safety net. If the Vision (CNN) model detects a malignant pattern with >40% confidence, it overrides the Meta-Learner's statistical bias (e.g., classifying as benign due to a patient's young age), preventing fatal False Negatives.
- **Dynamic Feature Weighting:** Removed the hardcoded 0.8/0.2 (Image/Clinical) weight ratio. The XGBoost model now dynamically calculates feature importance based on actual data correlations.
- **Debug & Conflict Dashboard:** Added a side-by-side "CNN (Vision) vs. Hybrid (Meta-Learner)" comparison panel in the Streamlit UI to transparently alert clinicians when metadata alters the visual diagnosis.
- **Advanced Preprocessing:** Integrated a dynamic Gaussian Vignette filter (sigma=180) to maintain focus on the lesion while preserving peripheral details that hard-cropping used to destroy.

## [v4.0.0] - Diagnostic Calibration Update
###  Major Pivot
- **Recall (Sensitivity) Focus:** Rather than the model's overall success, priority was given to Recall success so that it would not "miss" critical cases such as melanoma and BCC.
- **Discriminative Diagnosis (Top-3):** The model now lists the three most probable diseases with their percentages instead of a single result.

## [v3.0.0] - The "Honest AI" Update (TTA & Robust Classification)
### Additions and Corrections
- **Dynamic Threshold Values:** The risk warning system was refined by setting thresholds of 18% for melanoma and 25% for BCC.
- **Hybrid Score Synchronisation:** The actual risk score, weighted by Image (80%) and Clinical History (20%), was successfully integrated into the Streamlit interface.
- **Interface Improvement:** Images were fixed at 350px; Original and Heatmap (Grad-CAM) were made more readable side-by-side.
- **Architectural Update:** Migrated to FastAPI lifespan architecture and optimised Grad-CAM layer focus (-2).

- **Architectural Shift:** Transitioned from Metric Learning (Triplet Loss/KNN) to a **Weighted Classification** paradigm. The previous retrieval system was prone to overfitting on majority classes. The new architecture prioritizes **Recall on malignant classes** over general accuracy.
- **Inference Strategy:** Replaced single-pass inference with **Test-Time Augmentation (TTA)**. The model now "looks" at the lesion from 3 different angles (Original, H-Flip, V-Flip) and averages the probabilities to ensure diagnostic stability.

### Added
- **Weighted Cross-Entropy Loss:** Implemented class weighting to penalize the model heavily for missing Melanoma cases, directly addressing the dataset imbalance.
- **Learning Rate Scheduler:** Integrated `StepLR` (Gamma 0.1) to refine weights at later epochs, preventing local minima stagnation.
- **Out-of-Distribution (OOD) Robustness:** Verified system capability to distinguish non-cancerous skin conditions (e.g., random rashes from web) as "Normal" with low risk scores, reducing false positives.

### Removed / Deprecated
- **KNN Retrieval Engine:** The vector database approach has been archived in favor of direct probabilistic classification.
- **Legacy Triplet Loss:** Removed from the active production pipeline to reduce complexity.

## [v2.6.0] - Inference Optimization and Dual-Engine Support
### Added
- Integrated ONNX Runtime to optimize MobileNetV3-Large performance,
  reducing inference latency for CPU-based local deployment.
- Developed a dual-engine architecture supporting switching between
  high-speed screening and explainable Grad-CAM analysis.


## [v2.5.0] - Scaling Intelligence (Large Backbone)
### Changed
- **Architectural Upgrade:** Switched the Vision Backbone from MobileNetV3-Small to **MobileNetV3-Large (V2)**.
- **High-Fidelity Embeddings:** Increased the feature vector dimension from 576 to **960**, capturing significantly more granular dermatological features.
- **Performance Boost:** Achieved a state-of-the-art **94.75% Recall@5** on the validation set (up from ~87%).

### Added
- **Robust Augmentation Pipeline:** Integrated aggressive data augmentation strategies (Gaussian Blur, Color Jitter, Random Rotation) specifically tuned to prevent overfitting in the larger parameter space.
- **Strict Data Splitting:** Implemented `torch.utils.data.Subset` logic to ensure zero data leakage between training (augmented) and validation (clean) sets.

## [Unreleased / Planned]
### Focused
- **Pivot:** Project roadmap realigned to focus purely on **AI Engineering & MLOps**. Mobile development milestones have been deprecated in favor of Dockerization, Cloud Deployment, and Inference Optimization (ONNX).

## [v2.4.0] - Engineering Excellence (CI/CD)
### Added
- *Automated Testing:* Integrated pytest framework with 4 comprehensive test suites (test_dataset.py, test_inference.py, test_model.py, test_nlp_model.py) covering data integrity, model inference, and NLP tokenization.
- *CI/CD Pipeline:* Configured GitHub Actions workflow (.github/workflows/python-app.yml) to automatically trigger unit tests on every push to the main branch.
- *Badge Integration:* Added dynamic "Tests: Passing" badge to README for real-time build status visibility.

## [v2.3.0] - Explainable AI (XAI)
### Added
- *Grad-CAM Integration:* Implemented Gradient-weighted Class Activation Mapping to visualize model attention.
- *Interpretability Layer:* Generated heatmaps are now overlaid on original dermoscopy images to highlight the specific lesion boundaries influencing the KNN retrieval decision.

## [v2.2.0] - Multimodal Late Fusion
### Added
- *NLP Backbone:* Integrated DistilBERT transformer to process patient anamnesis (free-text symptoms).
- *Hybrid Scoring:* Engineered a weighted Late Fusion mechanism that combines MobileNetV3 visual embeddings with DistilBERT semantic features for a unified diagnostic score.

## [v2.1.0] - End-to-End Multimodal CBIR Integration
### Added
- Architected a high-throughput REST API utilizing FastAPI for real-time embedding extraction and K-Nearest Neighbors (KNN) similarity matching.
- Engineered an interactive clinical frontend via Streamlit, enabling seamless image upload and instant diagnostic feedback.
### Changed
- Transitioned the core modeling paradigm from static softmax classification to a dynamic Content-Based Image Retrieval (CBIR) architecture using Triplet Margin Loss.

## [v2.0.0] - The CBIR Pivot
### Changed
- Major architectural shift from standard classification to a 
  Content-Based Image Retrieval (CBIR) pipeline using KNN and Triplet Loss.

## [v1.1.0] - MLOps & Pipeline Standardization
### Changed
- Completely refactored the legacy training loops into the PyTorch Lightning framework, enforcing strict modularity and training reproducibility.
- Restructured the repository into a production-ready `src/` directory format, decoupling inference, model training, and API logic.
### Added
- Integrated dynamic learning rate scheduling (`ReduceLROnPlateau`) and automated model checkpointing based on validation loss monitoring.

## [v1.0.0] - Baseline Vision & Data Engineering
### Added
- Established the foundational PyTorch datasets and dataloaders with robust data augmentation pipelines (ColorJitter, RandomRotation, Normalization).
- Addressed severe dataset class imbalance using Scikit-Learn class weighting strategies within the loss function.
- Benchmarked initial vision architectures, evaluating Baseline Linear models, Custom CNNs, and Transfer Learning protocols with ResNet18.