# Age-and-Gender-Detection
This project uses OpenCV and deep learning to perform real-time age and gender prediction from your webcam feed. It detects faces in live video, then classifies each face’s age range and gender using pre-trained Caffe models.
Sure! Here's a GitHub-style description (README snippet) for your age and gender detection project using a webcam:

---

# 🎯 Real-Time Age and Gender Detection Using OpenCV and Deep Learning

This Python project leverages OpenCV, Haar Cascade, and deep learning models to perform **real-time age and gender prediction** using your device's webcam. The system detects faces in the video stream and uses pre-trained Caffe models to estimate the age range and gender of the person.

## 🚀 Features

- 📷 Real-time face detection using Haar Cascade
- 🧠 Age and gender prediction using pre-trained Caffe models
- 💬 Displays predictions directly on the video feed
- 🔄 Flip-frame mirror effect for a natural webcam experience
- 🖼️ Easy-to-read annotations (gender and age range)

## 📁 Files Included

- `age_gender_detection.py` – Main script for real-time detection
- `deploy_age.prototxt` / `age_net.caffemodel` – Age prediction model files
- `deploy_gender.prototxt` / `gender_net.caffemodel` – Gender prediction model files

## 🛠 Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install dependencies with:
```bash
pip install opencv-python numpy
```

## 🧪 How It Works

1. Accesses the default webcam (`cv2.VideoCapture(0)`)
2. Detects faces using Haar Cascades
3. Extracts the detected face and preprocesses it into a blob
4. Predicts age and gender using Caffe models
5. Displays predictions in real-time with bounding boxes and labels

## 🏁 Run the Project

```bash
python age_gender_detection.py
```

Press `q` to quit the window.

## 📌 Model Sources

- Age & Gender Caffe Models: [Download from here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) *(or specify your own download link)*
