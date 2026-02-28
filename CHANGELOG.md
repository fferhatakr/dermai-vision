# Changelog

All notable architectural changes, pipeline upgrades, and model iterations for the Skin Cancer Detection project will be documented in this file. The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.
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