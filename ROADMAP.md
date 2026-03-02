# Technical Roadmap & Architectural Evolution

This document outlines the strategic progression of the Skin Cancer CBIR project. The roadmap has been realigned to focus on **AI Engineering, MLOps, and Scalable Backend Systems**, prioritizing deployment robustness over mobile application development.

### Milestone 1: Multimodal Diagnostics Engine (v2.2.0) (Completed ✅)
* **Focus:** Bridging the gap between visual analysis and patient anamnesis.
* **Deliverable:** Integration of the DistilBERT NLP pipeline into the Streamlit frontend to compute a real-time Hybrid Diagnostic Score using Late Fusion techniques.

### Milestone 2: Clinical Interpretability & XAI (v2.3.0) (Completed ✅)
* **Focus:** Establishing diagnostic transparency for medical professionals.
* **Deliverable:** Implementation of Grad-CAM and Saliency heatmaps to visually isolate and highlight the specific lesion regions that drive the KNN retrieval mechanism.

### Milestone 3: Engineering Excellence (CI/CD) (v2.4.0) (Completed ✅)
* **Focus:** Standardizing the deployment environment and testing protocols.
* **Deliverable:** Implementation of Pytest suites and GitHub Actions workflows for automated continuous integration to ensure code stability.

### Milestone 4: Scaling Intelligence (v2.5.0) (Completed ✅)
* **Focus:** Maximizing diagnostic accuracy with larger backbones.
* **Deliverable:** -  Upgrade Vision Backbone to **MobileNetV3-Large**.
  -  Expand Vector Embedding dimension to **960**.
  -  Achieve >94% Recall on the validation set.

### Milestone 5: Out-of-Distribution (OOD) Hardening (v2.6.0) (Pending)
* **Focus:** Ensuring system safety and robustness against invalid inputs.
* **Deliverable:** Development of an anomaly detection layer to automatically identify and reject non-dermatological images, preventing erroneous clinical predictions.

### Milestone 6: High-Dimensional Vector Search (v2.7.0) (Pending)
* **Focus:** Scaling the retrieval database for production-level inference.
* **Deliverable:** Migration from standard PyTorch KNN to **Meta's FAISS (Facebook AI Similarity Search)** to achieve sub-millisecond retrieval latency across a large-scale embedding database.

### Milestone 7: Containerization & Microservices (v3.0.0) (Priority)
* **Focus:** Creating a production-ready, reproducible environment.
* **Deliverable:** - Full **Dockerization** of the FastAPI backend service.
  - writing a `docker-compose.yml` to orchestrate the API and Vector Database services together.

### Milestone 8: Cloud-Native Deployment (v3.1.0)
* **Focus:** Global accessibility and scalability.
* **Deliverable:** Deployment of the containerized inference engine to cloud infrastructure (e.g., AWS ECS, Google Cloud Run or a VPS) with NGINX as a reverse proxy.

### Milestone 9: Inference Optimization (ONNX & Quantization) (v4.0.0)
* **Focus:** Maximizing throughput and minimizing latency on CPU-based servers.
* **Deliverable:** - Exporting the PyTorch models to **ONNX Runtime** format.
  - Applying **Dynamic Quantization** (reducing model size/latency) to serve requests faster without needing expensive GPU instances.