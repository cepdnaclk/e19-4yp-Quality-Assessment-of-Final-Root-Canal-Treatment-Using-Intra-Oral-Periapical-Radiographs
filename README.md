# 🦷 AI-Assisted Tool for Assessing Quality of Final Root Canal Treatment Using Dental Radiographs

This repository contains the implementation of a deep learning-based tool to assist in evaluating the quality of root canal treatments (RCTs) using intraoral periapical (IOPA) radiographs. The tool aims to provide standardized, objective assessments that reduce diagnostic variability among dental practitioners.

## 📌 Project Overview

Root canal treatment (RCT) success is commonly assessed via IOPA radiographs. However, evaluations of treatment quality—including parameters like filling length, voids, and instrumentation errors—are often subjective and inconsistent.

This project proposes an AI-based tool that leverages deep learning techniques to:
- Automatically detect treated teeth
- Segment root canal fillings and key anatomical regions
- Assess treatment quality using clinically accepted metrics
- Provide visual and quantitative feedback through an interactive interface

---

## 🎯 Objectives

- 📸 Curate a dataset of 1,000 annotated IOPA radiographs
- 🤖 Develop machine learning models (CNN-based) for RCT classification
- 🧠 Segment treated areas and key features like root apex and pulpal floor
- 📊 Evaluate the AI tool against expert endodontists’ assessments
- 💻 Build a user-friendly web interface for clinical use

---

## 🧪 Methodology

- **Data Collection**: Anonymized IOPA radiographs from the Faculty of Dental Sciences
- **Annotation**: Experts annotate treated teeth, root filling, apex points, crowns, etc.
- **Preprocessing**: Image normalization, contrast enhancement, noise reduction
- **Model Pipeline**:
  - Treated Tooth Localization (e.g., CenterNet)
  - Instance & Semantic Segmentation (e.g., U-Net, Transformer-based models)
  - Keypoint Detection (e.g., root apex, pulpal floor)

  -  
    ![Methodology Diagram](docs/images/methodology.png)  
    *Figure: Overview of the methodology pipeline.*

- **Evaluation**: Accuracy, Precision, Recall, F1-Score, and AUC-ROC
- **Clinical Validation**: Comparison with endodontist assessments in trials

---

## 💡 Key Features Assessed

- ✅ Root canal filling length (underfilled / overfilled)
- ✅ Voids and filling compactness
- ✅ Separated instruments
- ✅ Coronal seal and crown quality
- ✅ Apex distance and pulpal floor integrity

---

## 🖥️ Interface

The tool will include a web-based interface built with:
- Backend: Flask or Django
- Frontend: React.js or Vue.js

Users can upload radiographs and receive:
- Real-time treatment quality predictions
- Visual overlays of segmentations and key findings

---

## 🧑‍💻 Team

- R.A.J.C. Adhikari – E/19/008  
- A.M.K.M. Adikari – E/19/009  
- D.M.G.S. Dassanayake – E/19/063  

**Supervisors**:
- Prof. R.D. Jayasinghe  
- Dr. I. Nawinne  
- Prof. R. Ragel  
- Dr. R.M.S.G.K. Rasnayake  
- Prof. R.M. Jayasinghe  
- Prof. M.C.N. Fonseka  
- Dr. P.A. Hewage  

---

## 📚 Reference

> Faculty of Engineering,  
> Department of Computer Engineering,  
> University of Peradeniya,  
> Final Year Research Project 2025

---

