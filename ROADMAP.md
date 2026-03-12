# Technical Roadmap & Architectural Evolution

This document outlines the strategic progression of the Skin Cancer CBIR project. The roadmap has been realigned to focus on **AI Engineering, MLOps, and Scalable Backend Systems**, prioritizing deployment robustness over mobile application development.
##  Phase I: Foundation & Research (Completed)
### Milestone 1: Multimodal Diagnostics Engine  (Completed )
* **Focus:** Bridging the gap between visual analysis and patient anamnesis.
* **Deliverable:** Integration of the DistilBERT NLP pipeline into the Streamlit frontend to compute a real-time Hybrid Diagnostic Score using Late Fusion techniques.

### Milestone 2: Clinical Interpretability & XAI  (Completed)
* **Focus:** Establishing diagnostic transparency for medical professionals.
* **Deliverable:** Implementation of Grad-CAM and Saliency heatmaps to visually isolate and highlight the specific lesion regions that drive the decision mechanism.

### Milestone 3: Engineering Excellence (CI/CD)  (Completed)
* **Focus:** Standardizing the deployment environment and testing protocols.
* **Deliverable:** Implementation of Pytest suites and GitHub Actions workflows for automated continuous integration to ensure code stability.

### Milestone 4: Scaling Intelligence  (Completed)
* **Focus:** Maximizing diagnostic accuracy with larger backbones.
* **Deliverable:** - Upgrade Vision Backbone to **MobileNetV3-Large**.
  - Expand Vector Embedding dimension to **960**.
  - Achieve >94% Recall on the validation set.

### Milestone 5: Out-of-Distribution (OOD) Hardening  (Completed)
* **Focus:** Ensuring system safety against non-lesion images.
* **Deliverable:** The Classifier V2 successfully identifies web-crawled non-cancerous rashes as "Normal" (Specificity), proving it doesn't just "guess" based on dataset bias.

### Milestone 6: Vector Search Scaling (Deprecated)
* **Note:** *Since the project pivoted from Retrieval to Direct Classification, the need for a FAISS vector database is obsolete.*

### Milestone 7: "Clinical Champion" Upgrade (v4.0.0) (Completed)
* **Focus:** Capturing "unmissable" cases by scaling up the architecture (EfficientNet) and optimising the error function (Focal Loss).
* **Deliverable:**
  - The image architecture has been upgraded to EfficientNet-B3.
  - Input resolution has been increased to 300x300 (Eagle Eye Resolution).
  - With Focal Loss integration, the Melanoma Recall (sensitivity) rate has been brought to the ~80% band.
  - Top-3 Classifier Diagnosis logic and 80%/20% Hybrid scoring have been linked to the interface.

---

##  Phase II: Data Science Robustness & Metadata (Completed)
### Milestone 8: Data Hygiene & Scientific Validation
* **Focus:** Proving the model's reliability scientifically and preventing "Clever Hans" learning (memorizing artifacts).
* **Deliverable:** 
  - **K-Fold Cross Validation:** Implementing a 5-Fold Cross Validation pipeline to prove that results are not due to a lucky train/test split.
  - **Artifact Removal (Digital Shaving):** Application of OpenCV-based algorithms to remove hair, rulers, and dark corners from images before training.(Various attempts were made but abandoned.)
  - **Advanced Balancing:** Testing Oversampling (SMOTE) or Advanced Class Weighting to further improve Melanoma recall.

### Milestone 9: Clinical Metadata Fusion(Completed)
* **Focus:** Moving from "Image Classification" to "Patient Diagnosis".
* **Deliverable:**
  - **Fusion Layer:** Designing a concatenation layer that combines image features (EfficientNet) with tabular data (Age, Gender, Anatomic Site).
  - **Dynamic Risk Adjustment:** Integrating "Duration of Lesion" and "Bleeding History" into the final probability score.

---

##  Phase III: MLOps & Industrial Engineering

### Milestone 10: Containerization & Reproducibility (The "AI Engineer" Standard)
* **Focus:** Eliminating "It works on my machine" issues and preparing for cloud deployment.
* **Deliverable:**
  - **Dockerization:** Creating `Dockerfile` for both API (FastAPI) and UI (Streamlit) services.
  - **Docker Compose:** Orchestrating the entire stack (API + Frontend + Redis cache) with a single command.
  - **Environment Standardization:** Locking all dependencies strictly to ensure 100% reproducibility across different OS (Linux/Mac/Windows).



### Milestone 11: Industrial Scaling & Modern Architectures
* **Focus:** Breaking the limits of local hardware and testing Next-Gen architectures.
* **Deliverable:**
  - **Distributed Training (DDP):** Implementing PyTorch Distributed Data Parallel to train on multi-GPU Cloud instances (AWS/GCP/Colab Pro).
  - **Next-Gen Benchmarks:** Training **Vision Transformers (ViT)** and **ConvNeXt** models to compare against EfficientNet.
  - **Ensemble Learning:** Combining the top 3 models (e.g., EfficientNet + ViT + ResNet) using a "Soft Voting" mechanism for maximum stability.

---

##  Phase IV: Optimization & Production Lifecycle

### Milestone 12: High-Performance Inference
* **Focus:** Reducing latency to millisecond levels for real-time usage.
* **Deliverable:**
  - **TensorRT Optimization:** Converting models to TensorRT engines (specifically for NVIDIA GPUs) to unlock 10x inference speed.
  - **Batch Inference Strategy:** Refactoring the API to handle multiple image uploads simultaneously (Batch Processing) efficiently.
  - **ONNX Runtime (CPU):** Fine-tuning ONNX export settings for maximum speed on standard CPUs (Edge Deployment).

### Milestone 13: System Observability & Monitoring
* **Focus:** Day-2 Operations – Keeping the model healthy in production.
* **Deliverable:**
  - **Drift Detection:** Implementing a pipeline (using tools like Evidently AI or Arize) to detect if incoming user images are significantly different from training data (Data Drift).
  - **Performance Monitoring:** Logging API latency, error rates, and model confidence distributions over time.

### Milestone 14: Medical UI/UX Transformation
* **Focus:** A scannable and reliable presentation of diagnostic results for doctors.
* **Deliverable:** - Redesign of the Streamlit interface to meet clinical standards (dark theme, clear metrics, comparative view).
  - Side-by-side comparison view for Ensemble model disagreements.



