# Changelog

All notable architectural changes, pipeline upgrades, and model iterations for the Skin Cancer Detection project will be documented in this file. The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [v2.3.0] - Async LLM Pipeline & UI Refactor

### Added
- Dedicated `/report` endpoint separating LLM generation from vision inference.
- Two-phase analysis flow: visual results render immediately, clinical report loads asynchronously.
- Step-by-step progress bar during Detailed Analysis.
- AI Clinical Report section with full Markdown rendering.
- Modular UI architecture split across 6 components.

## [v2.2.0] -LLM Clinical Decision Support
### Added
- Integrated LLM-driven CDSS to generate professional-grade clinical summaries.
- Implemented personalized patient routing and medical guidance protocols.
- Synthesized vision model predictions with clinical metadata for holistic reporting.


## [v2.1.0] - Fine-Tuning and Simulation
### Added
- DERM12345 Fine-Tuning Pipeline with mode switching capability.
- HSV Hair Removal using OpenCV TELEA inpainting.
- Hair Simulation Augmentation to improve robustness against artifacts.
- Dataset overlap validation and cross-dataset evaluation pipelines.

## [v2.0.0] - MIDAS Vision Backbone Upgrade
### Major Changes
- Upgraded core vision backbone to EfficientNet-B3 (MIDAS).
- Exported inference engine to ONNX format for optimized CPU execution.
### Added
- Retrained XGBoost Meta-Learner based on CNN out-of-fold predictions.
- Integrated ElasticTransform, GridDistortion, and AdvancedBlur augmentations.

## [v1.2.0] - Three-Tier Analysis and MLOps
### Added
- Three-Tier Analysis System: Quick Scan, Standard Analysis, and Detailed Analysis.
- MLflow Experiment Tracking for automated logging.
- Out-of-Distribution Detection guard using YOLO.

## [v1.1.0] - Containerization and Architecture Refactor
### Added
- Dockerized multi-container deployment architecture.
- Automated testing suites (Pytest) and GitHub Actions CI/CD pipeline.
### Fixed
- Resolved data leakage by implementing StratifiedGroupKFold.
- Corrected metric contamination between training and validation states.

## [v1.0.0] - Production Clinical System Integration
### Major Changes
- Integrated PostgreSQL database for patient and analysis management.
- Implemented JWT authentication for secure clinical endpoints.
- Refactored API into modular components (routes, schemas, models).

## [v0.4.0] - Meta-Learner Architecture
### Major Changes
- Replaced NLP DistilBERT architecture with an XGBoost Meta-Learner for structured clinical metadata.
- Implemented a clinical safety override logic prioritizing vision model confidence on malignant patterns.

## [v0.3.1] - Diagnostic Calibration
### Changed
- Prioritized Recall (Sensitivity) optimization over general accuracy.
- Updated output generation to provide Top-3 discriminative diagnoses.

## [v0.3.0] - Weighted Classification Pivot
### Major Changes
- Shifted from Content-Based Image Retrieval (CBIR) to a Weighted Classification architecture.
- Replaced single-pass inference with Test-Time Augmentation (TTA).
### Added
- Integrated class weighting to penalize false negatives for malignant classes.

## [v0.2.1] - Backbone Scaling and Explainability
### Changed
- Upgraded Vision Backbone from MobileNetV3-Small to MobileNetV3-Large.
### Added
- Integrated Grad-CAM for Explainable AI (XAI) visualizations.
- Implemented dual-engine architecture supporting ONNX optimization.

## [v0.2.0] - Multimodal NLP Integration
### Major Changes
- Shifted from standard classification to a Content-Based Image Retrieval (CBIR) pipeline.
### Added
- Integrated DistilBERT transformer for processing patient anamnesis.
- Engineered a Late Fusion mechanism combining visual embeddings and semantic features.
- Implemented FastAPI backend and Streamlit frontend for clinical interactions.

## [v0.1.0] - Baseline Vision Pipeline
### Added
- Established foundational datasets, dataloaders, and augmentation pipelines.
- Benchmarked initial vision architectures including ResNet18.
- Refactored legacy loops into the PyTorch Lightning framework.