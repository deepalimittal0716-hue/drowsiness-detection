# Project Report

## Face \& Eye Detection System

### BYOP — Computer Vision Capstone

**Student Name:** Deepali Mittal

**Registration No.:** 23BAI10645

**Slot:**F11+F12
**Course:** Computer Vision
**Submission Date: 31** March 2026

\---

## 1\. Problem Statement

In many real-world scenarios, there is a need to automatically detect and locate the presence of human faces in images. This is particularly true in classrooms, offices, cars, and security systems.

In one real-world application, for example, it is possible to build a system to detect whether a driver is awake or asleep by checking whether his eyes are open or closed. It is also possible to build a system to detect the number of people in a frame for an attendance system.

In this project, the aim is to build a computer vision system that is able to:

* Detect all the human faces in an image
* Detect the eyes within the detected faces
* Draw bounding boxes around the detected faces and eyes
* Save the output image

\---

## 2\. Why This Problem Matters

Face and eye detection is a fundamental building block for a variety of high-impact applications:

* **Road Safety:** Drowsiness detection cameras in cars check for open eyes of drivers. Companies like Bosch and Mobileye use such technology.
* **Attendance Automation:** Educational institutions can use face detection to automate attendance for their students by detecting faces in camera feeds of classrooms.
* **Security:** Security cameras can send alerts for unauthorized faces in a restricted area.
* **Healthcare:** Alertness or eye gaze of patients in ICUs can also be monitored using face and eye detection technology.

A simple face and eye detection system will show how all these high-impact applications work

\---

# 3\. Approach and Methodology

### 3.1 Technology Chosen

I chose OpenCV's Haar Cascade Classifiers for this project because:

* They are already built into OpenCV and do not require the model to be downloaded separately
* They do not require a GPU or deep learning setup
* They are very fast and efficient for real-time applications
* They are direct applications of concepts learned in the course

## 3.2 How Haar Cascade Classifiers Work

Haar Cascades use a sliding window approach:

* Scanning the image at various scales (image pyramid)
* At each location, it checks if it looks like a face
* It checks by using a collection of simple rectangular regions (Haar features) to vote
* Regions must pass all checks to be classified as a face

This is called the **Viola-Jones algorithm (2001)**, one of the first face detection systems in real-time.

## 3.3 Processing Pipeline

```
Input Image
    ↓
Convert to Grayscale
    ↓
Apply Face Cascade (detectMultiScale)
    ↓
For each face → Extract ROI
    ↓
Apply Eye Cascade on ROI
    ↓
Draw Bounding Boxes
    ↓
Save Output Image
```

### 3.4 Key Parameters Tuned

|Parameter|Value Used|Reason|
|-|-|-|
|`scaleFactor`|1.1|10% scale reduction per step — good trade-off|
|`minNeighbors`|5 (face), 10 (eye)|Higher value for eye detection to avoid false positives|
|`minSize`|(30,30) face, (15,15) eye|Ignore noisy regions that are too small|

\---

## 4\. Implementation

The project was implemented using the Python programming language with the following libraries:

* **`cv2` -** OpenCV for image processing
* **`numpy` -** for array operations
* **`matplotlib` -** for image display in Google Colab
* **`os`, `sys`** - for file handling

Two versions of the project have been implemented:

1. **Google Colab Notebook (`Face\_Eye\_Detection.ipynb`) -** interactive, step-by-step with explanations
2. **Standalone Python Script (`face\_eye\_detection.py`) -** command-line application for processing individual images or entire folders

\---

## 5\. Key Decisions Made

**Decision 1: Using ROI in Eye Detection**

Rather than scanning the whole image for eyes, I decided to first detect the face and then look for eyes within the face region of interest. This greatly reduces false alarms because the algorithm might otherwise recognize other patterns in clothing or the background as eyes.

**Decision 2: Different MinNeighbors for Faces and Eyes**

For faces, I set **`minNeighbors=5**`, and for eyes, I set **`minNeighbors=10`**. This is because the eyes are much smaller and the classifier is more prone to false alarms, requiring a stricter criterion.

**Decision 3: Refactoring the Code into a Function**

Rather than writing the whole code in one go, I decided to break it into a function called **`detect\_faces\_and\_eyes()`**. This not only makes the code more reusable but also allows it to be more easily tested against other images.

\---

## 6\. Challenges Faced

**Challenge 1: False Positives**

Early images contained false positives, i.e., "faces" in background images, shadows, and texture. Solution: Increase `minNeighbors` and `minSize`.

**Challenge 2: Side Profile Faces Not Detected**

I realized that the **`haarcascade\_frontalface\_default.xml`** is trained to detect frontal faces. Side-profile faces were not detected. This is one limitation of the Haar cascades. A profile cascade **(`haarcascade\_profileface.xml`)** or a deep learning-based method would be required to solve this challenge.

**Challenge 3: Low-Light Images**

Accuracy in images taken in low light is not high. **Improvement for the future:** Use histogram equalization **(`cv2.equalizeHist`)** prior to object detection.

**Challenge 4: Eye detection inside partially visible faces**

When the face is near the edge of the image, the ROI is not correctly defined, resulting in failure to detect the eyes. **Solution:** Ensure the ROI coordinates are within the image boundaries.---

## 7\. Results

The system was tested on some images:

* ✅ **Frontal pose of 1 person with good lighting:** 1 face, 2 eyes correctly detected
* ✅ **Group of 4-5 people:** Most faces detected
* ⚠️ **Side profile:** Not detected (known issue of Haar cascade)
* ⚠️ **Person wearing sunglasses:** Eyes not detected (because of sunglasses)
* ✅ **Close-up of 1 person:** Correctly detected

The experiment in the parameter comparison section (Step 6 of the notebook) demonstrated how the sensitivity and precision of the detector depend on the choice of `scaleFactor` and `minNeighbors`.---

## 8\. What I Learned

* Why is **Grayscale conversion** important?
* Why is **Scale invariance** important?
* **Region of Interest** is a very effective method to reduce the scope to specific areas, which helps in reducing errors and computation time.
* Why is **parameter tuning** important?
* **Haar Cascades** are not perfect; they are fast and light, but not as effective as other state-of-the-art detectors like MTCNN, YOLO, etc.

\---

## 9\. Future Improvements

|**Improvement**|**How**|
|-|-|
|Real-time webcam detection|Use **`cv2.VideoCapture(0)`** loop|
|Better accuracy|Use **Dlib** or **MTCNN** for deep learning-based detection|
|Emotion recognition|Add a **CNN** trained on facial expressions|
|Profile face detection|Add **`haarcascade\_profileface.xml`**|
|Low-light support|Apply **`cv2.equalizeHist()`** before detection|

\---

## 10\. Conclusion

This project has successfully shown a face and eye detection system using classical computer vision techniques. The techniques used by this face and eye detection system are concepts from this course, including image processing, feature detection, and cascade classifiers.

Even though this face and eye detection system has many shortcomings compared to modern techniques, it is a fully CPU-based solution that does not require any training data and works out of the box using OpenCV.

\---

## References

1. Viola, P., \& Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *CVPR 2001*.
2. OpenCV Documentation: https://docs.opencv.org/
3. OpenCV Haar Cascades: https://github.com/opencv/opencv
4. WHO Global Status Report on Road Safety (2023)

