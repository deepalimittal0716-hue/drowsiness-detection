"""
Driver Drowsiness Detection System
-----------------------------------
Detects faces and eyes in an image using OpenCV Haar Cascades.
Determines if eyes are open or closed and outputs a drowsiness status.

Usage:
    Single image:  python drowsiness_detection.py image.jpg
    Folder:        python drowsiness_detection.py input_folder/ output_folder/
"""

import cv2
import sys
import os

# Load Haar Cascade classifiers (built into OpenCV, no download needed)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

DROWSY_THRESHOLD = 3  # score at which alert is triggered

def analyze_image(image_path, output_path=None):
    """
    Analyze a single image for drowsiness.
    Returns: status string, eye count, drowsiness score
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None, 0, 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        print(f"[!] No face detected in: {image_path}")
        return "NO_FACE", 0, 0

    drowsy_score = 0
    total_eyes = 0
    status = "AWAKE"

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 100, 0), 2)
        cv2.putText(img, "Face", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        # Region of interest - upper 60% of face for eyes
        roi_h = int(h * 0.6)
        roi_gray  = gray[y:y+roi_h, x:x+w]
        roi_color = img[y:y+roi_h, x:x+w]

        # Detect eyes inside face region
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
        )
        eye_count = len(eyes)
        total_eyes += eye_count

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 200), 2)

        # Determine drowsiness score
        if eye_count >= 2:
            face_status = "EYES OPEN"
            box_color   = (0, 200, 100)
        elif eye_count == 1:
            drowsy_score += 1
            face_status  = "ONE EYE?"
            box_color    = (0, 165, 255)
            status       = "WARNING"
        else:
            drowsy_score += 2
            face_status  = "EYES CLOSED"
            box_color    = (0, 0, 255)
            status       = "DROWSY"

        cv2.putText(img, face_status, (x, y+h+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # Trigger alert if score high enough
    if drowsy_score >= DROWSY_THRESHOLD:
        status = "ALERT"
        cv2.rectangle(img, (0, 0), (img.shape[1], 45), (0, 0, 200), -1)
        cv2.putText(img, "!! DROWSINESS ALERT !! WAKE UP !!",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

    # Summary text
    cv2.putText(img, f"Eyes: {total_eyes}  Score: {drowsy_score}  Status: {status}",
                (10, img.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Save output
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = base + "_result" + ext
    cv2.imwrite(output_path, img)

    print(f"  Image   : {image_path}")
    print(f"  Eyes    : {total_eyes}")
    print(f"  Score   : {drowsy_score}")
    print(f"  Status  : {status}")
    print(f"  Saved   : {output_path}")
    print()

    return status, total_eyes, drowsy_score


def process_folder(input_folder, output_folder):
    """Process all images in a folder."""
    os.makedirs(output_folder, exist_ok=True)
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported)]

    if not files:
        print(f"No images found in {input_folder}")
        return

    print(f"\nProcessing {len(files)} image(s)...\n")
    for filename in files:
        in_path  = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, filename)
        analyze_image(in_path, out_path)

    print(f"Done! Results saved in: {output_folder}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and os.path.isdir(sys.argv[1]):
        process_folder(sys.argv[1], sys.argv[2])
    elif len(sys.argv) >= 2:
        analyze_image(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
    else:
        print("Usage:")
        print("  Single image : python drowsiness_detection.py image.jpg")
        print("  Folder       : python drowsiness_detection.py input/ output/")
