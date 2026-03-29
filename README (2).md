# Driver Drowsiness Detection System

A computer vision project that detects whether a person's eyes are open or closed using Python and OpenCV, and triggers a drowsiness alert when eyes remain closed for too long.

---

## Problem Statement

Drowsy driving causes thousands of road accidents every year. This project uses computer vision to monitor eye state and alert the driver when drowsiness is detected.

---

## Tech Stack

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib

---

## Project Structure

```
drowsiness-detection/
├── Drowsiness_Detection.ipynb   # Google Colab notebook (recommended)
├── drowsiness_detection.py      # Command-line Python script
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── PROJECT_REPORT.md            # Project report
└── results/                     # Sample output images
```

---

## How to Run

### Option 1: Google Colab (Easiest)

1. Go to https://colab.research.google.com
2. Click File → Upload notebook
3. Upload `Drowsiness_Detection.ipynb`
4. Click Runtime → Run all
5. Upload your own image when prompted in Step 5

### Option 2: Command Line (Local Machine)

#### Step 1: Make sure Python is installed
```bash
python --version
```
If not installed, download from https://www.python.org/downloads/

#### Step 2: Clone the repository
```bash
git clone https://github.com/deepalimittal0716-hue/drowsiness-detection
cd drowsiness-detection
```

#### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Run on a single image
```bash
python drowsiness_detection.py path/to/your/image.jpg
```

#### Step 5: Run on a folder of images
```bash
python drowsiness_detection.py input_folder/ output_folder/
```

---

## How It Works

1. Load image and convert to grayscale
2. Detect face using Haar Cascade classifier
3. Crop the upper 60% of the face (Region of Interest)
4. Detect eyes inside that region
5. Count eyes: 2 = AWAKE, 1 = WARNING, 0 = DROWSY
6. Update drowsiness score — if score exceeds threshold, trigger ALERT

---

## Sample Output

- Blue box = detected face
- Cyan boxes = detected eyes
- Red banner = DROWSINESS ALERT triggered

---

## Limitations

- Works best with frontal, well-lit faces
- Does not work with sunglasses
- Haar Cascade does not detect side-profile faces

---

## Author

Deepali Mittal - 23BAI10645
Computer Vision Course — BYOP Capstone
March 2026
