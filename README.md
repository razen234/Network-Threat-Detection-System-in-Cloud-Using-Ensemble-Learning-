# 🌐 Network Threat Detection System in Cloud Using Ensemble Learning

This project presents a cloud-ready **Network Threat Detection System** that uses **Ensemble Machine Learning Models** to detect various cyber-attacks in network traffic. It is designed to support real-time or batch traffic analysis and provides a user-friendly **Flask Web Application** for uploading CSV traffic files or manually entering packet values.

The system is based on the **CIC-IDS 2018** dataset and uses a combination of classical ML algorithms such as Random Forest, AdaBoost, Naive Bayes, and a Voting Ensemble. The final trained models are stored using **Git LFS** because of their large size.

---

## 🚀 Key Features

### 🔐 **Threat Classification Models**
- Random Forest  
- AdaBoost  
- Gaussian Naive Bayes  
- Voting Classifier (Majority Vote)

### 📊 **Preprocessing Pipeline**
- Missing value handling  
- Normalization  
- Feature scaling  
- Feature selection (optional)  
- Label encoding  

### 🧠 **Ensemble Learning**
Combines multiple models to:
- Improve accuracy  
- Reduce false positives  
- Offer stronger generalization  

### 🌍 **Cloud-Ready System**
- Supports large CSV uploads  
- Model files stored using Git LFS  
- Flask backend for API predictions  
- Lightweight UI for quick detection  

---

## 🖥️ System Architecture

