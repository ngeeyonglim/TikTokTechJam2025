# PhotoCensor: Privacy-Preserving Photo Gallery  

A photo gallery web application that empowers users to **censor sensitive content** in their images while contributing to **privacy-preserving AI training**.  

Users can blur faces or other regions, manually draw bounding boxes around uncensored features, and automatically generate new training samples. These edits improve a local model (PyTorch + YOLO), which is then aggregated using **Federated Learning (FL)** and **Differential Privacy (DP)**.

---

## 📖 Table of Contents  
- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
---

## 🔍 Overview  
Most photo apps don’t guarantee privacy. **PhotoCensor** puts users in control:  

- Automatically detect and censor sensitive regions in images.  
- Manually draw bounding boxes to blur any missed features.  
- Save censored images locally.  
- Improve the on-device model through federated updates — without ever sending raw images to a server.  
- Share updates securely using differential privacy, ensuring personal data remains protected.  

---

## ✨ Features  
- **📸 Photo Gallery UI**: Browse and select images in a familiar interface.  
- **🤖 Automated Censoring**: Detects and blurs sensitive regions.  
- **✏️ Manual Editing**: Draw bounding boxes for uncensored features.  
- **🧠 On-device Learning**: Add training samples seamlessly through your edits.  
- **🌍 Federated Learning**: Contribute to a global model without exposing private data.  
- **🔒 Differential Privacy**: Ensure updates are mathematically private.  

---

## ⚙️ Installation  

### Prerequisites
- Python 3.10+  
- Node.js 18+  
- pip / virtualenv  
- Git LFS (if you want to manage large models)  

### Backend (Flask + PyTorch)
```bash
# clone repo
git clone https://github.com/<username>/<repo>.git
cd <repo>

# setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install Python dependencies
pip install -r requirements.txt
```

### Usage
<Youtube Link>

