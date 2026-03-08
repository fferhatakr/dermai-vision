# Technical Roadmap & Architectural Evolution

This document outlines the strategic progression of the Skin Cancer CBIR project. The roadmap has been realigned to focus on **AI Engineering, MLOps, and Scalable Backend Systems**, prioritizing deployment robustness over mobile application development.

### Milestone 1: Multimodal Diagnostics Engine (v2.2.0) (Completed )
* **Focus:** Bridging the gap between visual analysis and patient anamnesis.
* **Deliverable:** Integration of the DistilBERT NLP pipeline into the Streamlit frontend to compute a real-time Hybrid Diagnostic Score using Late Fusion techniques.

### Milestone 2: Clinical Interpretability & XAI (v2.3.0) (Completed )
* **Focus:** Establishing diagnostic transparency for medical professionals.
* **Deliverable:** Implementation of Grad-CAM and Saliency heatmaps to visually isolate and highlight the specific lesion regions that drive the KNN retrieval mechanism.

### Milestone 3: Engineering Excellence (CI/CD) (v2.4.0) (Completed )
* **Focus:** Standardizing the deployment environment and testing protocols.
* **Deliverable:** Implementation of Pytest suites and GitHub Actions workflows for automated continuous integration to ensure code stability.

### Milestone 4: Scaling Intelligence (v2.5.0) (Completed )
* **Focus:** Maximizing diagnostic accuracy with larger backbones.
* **Deliverable:** -  Upgrade Vision Backbone to **MobileNetV3-Large**.
  -  Expand Vector Embedding dimension to **960**.
  -  Achieve >94% Recall on the validation set.

### Milestone 5: Out-of-Distribution (OOD) Hardening (v3.0.0) (Completed )
* **Focus:** Ensuring system safety against non-lesion images.
* **Deliverable:** The Classifier V2 successfully identifies web-crawled non-cancerous rashes as "Normal" (Specificity), proving it doesn't just "guess" based on dataset bias.

### Milestone 6: Vector Search Scaling (Deprecated)
* **Note:** *Since the project pivoted from Retrieval to Direct Classification, the need for a FAISS vector database is obsolete.*

### Milestone 7: "Clinical Champion Upgrade (v4.0.0) (Completed )
* **Focus:** Capturing "unmissable" cases by scaling up the architecture (EfficientNet) and optimising the error function (Focal Loss).
* **Deliverable:**
  - The image architecture has been upgraded to EfficientNet-B3.
  - Input resolution has been increased to 300x300 (Eagle Eye Resolution).
  - With Focal Loss integration, the Melanoma Recall (sensitivity) rate has been brought to the 80% band.
  - Top-3 Classifier Diagnosis logic and 80%/20% Hybrid scoring have been linked to the interface.

### Milestone 8: Data Hygiene & Digital Shaving
* **Focus:** Prevent the model from memorising secondary factors such as the ruler, hair or black frame (Clever Hans Effect).
* **Deliverable:** 
  - Application of the OpenCV-based Artifact Removal (Digital Shaving) algorithm to the training set.
  - Analysis from three different angles during inference with TTA (Test Time Augmentation) integration.

### Milestone 9: Metadata Fusion & Clinical Context
* **Focus:** Not just based on the picture, but making a diagnosis according to the patient's profile.
* **Deliverable:**
  - Fusion Layer: A new neural network layer that combines age, gender and body region (CSV data) information with image features.
  - The duration of the lesion on the body and the inclusion of skin type in the hybrid score.

### Milestone 10:  Industrial Scaling
* **Focus:** Going beyond the limits of the RTX 3050 to specialise in big data.
* **Deliverable:**
  - Training between 60,000 and 100,000 images (ISIC Full Archive) on Cloud Servers (AWS/Colab).
  - Benchmarking deeper architectures such as EfficientNet-B4 or Inception v3.

### Milestone 11: Optimization & Ensemble
* **Focus:** Balance between speed and maximum accuracy.
* **Deliverable:**
  - Ensemble Learning: Combining the three best-performing models (B3 + B4 + ResNet) using a voting method.
  - ONNX Optimisation: Reducing the API response time to the millisecond level.

### Milestone 12: Medical UI/UX Transformation
* **Focus:** A scannable and reliable presentation of diagnostic results for doctors.
* **Deliverable:** Redesign of the Streamlit interface to meet clinical standards (dark theme, clear metrics, comparative view).


*
