# Technical Roadmap & Architectural Evolution

This document outlines the strategic progression of the Skin Cancer CBIR project. The roadmap is designed to scale the current prototype into a robust, interpretable, and edge-deployable clinical AI assistant, adhering strictly to MLOps and software engineering best practices.

### Milestone 1: Multimodal Diagnostics Engine (v2.2.0)
* **Focus:** Bridging the gap between visual analysis and patient anamnesis.
* **Deliverable:** Integration of the DistilBERT NLP pipeline into the Streamlit frontend to compute a real-time Hybrid Diagnostic Score using Late Fusion techniques.

### Milestone 2: Clinical Interpretability & XAI (v2.3.0)
* **Focus:** Establishing diagnostic transparency for medical professionals.
* **Deliverable:** Implementation of Grad-CAM and Saliency heatmaps to visually isolate and highlight the specific lesion regions that drive the KNN retrieval mechanism.

### Milestone 3: Out-of-Distribution (OOD) Hardening (v2.4.0)
* **Focus:** Ensuring system safety and robustness against invalid inputs.
* **Deliverable:** Development of an anomaly detection layer to automatically identify and reject non-dermatological images, preventing erroneous clinical predictions.

### Milestone 4: High-Dimensional Vector Search (v2.5.0)
* **Focus:** Scaling the retrieval database for production-level inference.
* **Deliverable:** Migration from standard PyTorch KNN to Meta's FAISS (Facebook AI Similarity Search) to achieve sub-millisecond retrieval latency across a database of 100,000+ embedding vectors.

### Milestone 5: Continuous Integration & Containerization (v3.0.0)
* **Focus:** Standardizing the deployment environment and testing protocols.
* **Deliverable:** Full Dockerization of the FastAPI backend and Streamlit frontend, coupled with GitHub Actions workflows for automated unit testing (CI/CD).

### Milestone 6: Cloud-Native API Gateway (v3.1.0)
* **Focus:** Global accessibility and concurrent request handling.
* **Deliverable:** Deployment of the containerized inference engine to cloud infrastructure (e.g., AWS ECS or Google Cloud Run) with proper load balancing.

### Milestone 7: Edge AI & Inference Optimization (v4.0.0)
* **Focus:** Decoupling the system from heavy cloud dependencies.
* **Deliverable:** Exporting the MobileNetV3 vision backbone to ONNX format and applying INT8 Post-Training Quantization to reduce the model footprint below 5MB without significant mAP degradation.

### Milestone 8: Offline Mobile Application (v4.1.0)
* **Focus:** End-user accessibility in remote clinical environments.
* **Deliverable:** Development of a native iOS application using Swift and CoreML to run the quantized hybrid diagnostic engine entirely on-device with zero latency.