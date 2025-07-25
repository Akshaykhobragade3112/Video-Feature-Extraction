# 🎥 Video Feature Extraction Tool 

This project is a Python-based tool that extracts key **visual** and **temporal** features from local video files using computer vision and deep learning techniques.

It was built as part of the **White Panda Machine Learning Engineer assignment**, focusing on core video analysis without relying on OCR.

---

## 🚀 Features Extracted

The tool processes videos and computes the following three features:

### 1️⃣ Shot Cut Detection
- Detects **"hard cuts"** between scenes using histogram difference between consecutive frames.
- Helps identify transitions or scene changes in a video.

### 2️⃣ Motion Analysis
- Measures **average motion** in the video using **Optical Flow (Farneback method)**.
- Quantifies how much the content in the video is moving overall.

### 3️⃣ Object vs. Person Ratio
- Uses a pre-trained **YOLOv8** model to detect and classify people vs other objects.
- Calculates the **ratio of people to objects** across sampled frames:



---

## 📁 Project Structure

Video Feature Extraction/
├── Videos/ # 📂 Input videos (.mp4)
├── VideoFeatureExtraction.py # 🧠 Main script
├── requirements.txt # 📦 Python dependencies
└── README.md # 📘 Project documentation
---

## 📥 Setup Instructions

### 1. Install Python
Make sure you have Python 3.8+ installed.  
Check with:
```bash
python --version


## Install Dependencies
Run this command inside the project folder:
   pip install -r requirements.txt
