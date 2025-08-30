# hide & seek: A Privacy-first Photo Gallery

A photo gallery application that empowers users to **censor sensitive content** in their images while contributing to **privacy-preserving AI training**.  

Users can blur faces or other regions, manually draw bounding boxes around features, which automatically generate new training data. These edits improve a local model (PyTorch + YOLO), which is then aggregated using **Federated Learning (FL)** and **Differential Privacy (DP)** to train a global model.

---

## 📖 Table of Contents  
- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Contributing](#contributing)  
- [Roadmap](#roadmap)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)  

---

## 🔍 Overview  
Most photo apps don’t guarantee privacy. **Hide&Seek** puts users in control:  

- Automatically detect and censor sensitive regions in images.  
- Manually draw bounding boxes to blur any missed features.  
- Save censored images locally.  
- Improve the on-device model through federated updates — without ever sending raw images to a server.  
- Share updates securely using differential privacy, ensuring personal data remains protected.  

---

## ✨ Features  
- **📸 Photo Gallery UI**: Browse and select images in a familiar interface.  
- **🤖 Censoring**: Detects and blurs sensitive regions.  
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
pip install -r ml_code/requirements.txt
```

### Frontend (Lynx.js)
```bash
cd frontend
npm install 
npm run build
```

---

## 🎥 Demo  
👉 (https://youtu.be/3w1zr4m3QPs)

---

## 🤝 Contributing  
Contributions are welcome!  
1. Fork the repo  
2. Create a branch (`git checkout -b feature/xyz`)  
3. Commit your changes (`git commit -m "Add feature xyz"`)  
4. Push the branch (`git push origin feature/xyz`)  
5. Open a Pull Request  

---

## 🛣️ Roadmap  
- [ ] Add mobile app version (React Native or Lynx mobile)  
- [ ] Improve YOLO detection accuracy  
- [ ] Add support for video censoring  
- [ ] UI for federated training progress  
- [ ] Deployment on cloud FL coordinator  

---

## 📜 License  
This project is licensed under the **MIT License** — free to use, modify, and distribute.  

---

## 🙏 Acknowledgments  
- [Flask](https://flask.palletsprojects.com/) – Backend framework  
- [Lynx.js](https://lynx.tiktokglobal.lan/) – TikTok frontend framework  
- [PyTorch](https://pytorch.org/) – ML training  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – Object detection  
- [Opacus](https://opacus.ai/) – Differential privacy in PyTorch  
- [Flower](https://flower.dev/) – Federated learning framework  

---
