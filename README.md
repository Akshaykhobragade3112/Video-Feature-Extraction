#  Video Feature Extraction Tool 

This project is a Python-based tool that extracts key **visual** and **temporal** features from local video files using computer vision and deep learning techniques.

It was built as part of the **White Panda Machine Learning Engineer assignment**, focusing on core video analysis without relying on OCR.

---

##  Features Extracted

The tool processes videos and computes the following three features:

### 1 Shot Cut Detection
- Detects **"hard cuts"** between scenes using histogram difference between consecutive frames.
- Helps identify transitions or scene changes in a video.

### 2 Motion Analysis
- Measures **average motion** in the video using **Optical Flow (Farneback method)**.
- Quantifies how much the content in the video is moving overall.

### 3 Object vs. Person Ratio
- Uses a pre-trained **YOLOv8** model to detect and classify people vs other objects.
- Calculates the **ratio of people to objects** across sampled frames:



---

##  Project Structure
```bash
Video Feature Extraction/  
├── Videos/ #  Input videos (.mp4)  
├── VideoFeatureExtraction.py #  Main script    
├── requirements.txt #  Python dependencies    
└── README.md #  Project documentation
```
---

##  Setup Instructions

### 1. Install Python
Make sure you have Python 3.8+ installed.  
Check with:
```bash
python --version
```

### 2. Install Dependencies
Run this command inside the project folder:
```bash
pip install -r requirements.txt
```

### 3. Add Videos
Place your .mp4 files inside the Videos/ folder. Example names:
```bash
SampleVideo1.mp4

SampleVideo2.mp4
```
### 4. Run the Script
```bash
python VideoFeatureExtraction.py
```

## How it works
Frame Sampling: Every 5th frame is extracted for faster processing

Hard Cuts: Detected using grayscale histogram difference

Motion: Optical Flow is calculated between consecutive frames

Object Detection: YOLOv8 detects objects and persons in sampled frames

## requirements.txt
```bash
opencv-python
numpy
ultralytics
torch
```

## Sample Output
```bash
{
  "hard_cuts": 3,
  "average_motion": 0.42,
  "person_object_ratio": 1.75
}
```
