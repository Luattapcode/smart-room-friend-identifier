# 🚪 Smart Room Friend Identifier (AI Biometric Access)

A lightweight, real-time facial recognition and room visitor identification system built with **Python**, **OpenCV**, and Deep Learning models (**YuNet** & **SFace**).

Runs 100% locally on standard CPU without requiring cloud APIs or GPU setup.

---

## ✨ Key Features
- **Real-Time Detection:** Powered by OpenCV's ultra-lightweight CNN detector (**YuNet**), detecting faces and 5 facial landmarks at 30+ FPS.
- **Biometric Identification:** Uses **SFace** (Deep Metric Learning) to extract 128-dimensional facial embedding vectors and matches identities using Cosine Similarity.
- **Zero-Shot Registration (Hot Enroll):** Add new friends into the system on the spot by pressing `r` without restarting or retraining the AI model.
- **Privacy-Friendly:** Runs entirely offline on local hardware.

---

## 🧠 System Architecture

```text
[Webcam Frame] 
      │
      ▼
1. Face Detection (YuNet CNN) ──> Bounding box (x, y, w, h) + 5 landmarks
      │
      ▼
2. Face Alignment & Cropping  ──> Affine rotation to level eyes, normalized to 112x112
      │
      ▼
3. Feature Extraction (SFace) ──> 128-D Embedding vector: v ∈ ℝ¹²⁸
      │
      ▼
4. Cosine Similarity Matching ──> cos(θ) = (u · v) / (||u|| * ||v||)
      │
      ├─ Score >= 0.363 ──> Friend Verified (Green Box + Name)
      └─ Score <  0.363 ──> Unknown Guest   (Red Box)
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/smart-room-identifier.git
cd smart-room-identifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add initial friends (Optional)
Create a `friends/` folder and place clear portrait photos:
```text
friends/
├── alex.jpg
├── john.png
└── me.jpg
```
*(Filename will be used as the displayed name).*

### 4. Run the application
```bash
python main.py
```
*Models (`face_detection_yunet.onnx` and `face_recognition_sface.onnx`) will be automatically downloaded on first run.*

---

## ⌨️ Controls
| Key | Action |
|---|---|
| `r` | **Register:** Enroll the face currently seen on camera with a new name |
| `q` | **Quit:** Stop video feed and close application |