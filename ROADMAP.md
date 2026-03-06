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

### Milestone 5: Out-of-Distribution (OOD) Hardening (v3.0.0) (Completed ✅)
* **Focus:** Ensuring system safety against non-lesion images.
* **Deliverable:** The Classifier V2 successfully identifies web-crawled non-cancerous rashes as "Normal" (Specificity), proving it doesn't just "guess" based on dataset bias.

### Milestone 6: Vector Search Scaling (Deprecated ❌)
* **Note:** *Since the project pivoted from Retrieval to Direct Classification, the need for a FAISS vector database is obsolete.*

### Milestone 7: "Eagle Eye" Resolution Upgrade (v3.1.0) (Next Priority 🔥)
* **Focus:** Improving Melanoma Recall (sensitivity) by capturing micro-textures.
* **Deliverable:**
  - Increase input resolution from **224x224** to **448x448**.
  - Implement **Focal Loss** to force the model to focus on "hard-to-classify" malignant cases (the missing 16/33).

### Milestone 8: Transformer Era (v4.0.0) (Future Research)
* **Focus:** Global context understanding.
* **Deliverable:** Benchmarking **Vision Transformers (ViT)** or **ConvNeXt** against the current MobileNetV3 to see if attention mechanisms outperform CNNs on skin lesion borders.