import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from utils.utils import get_car, read_license_plate, write_csv
from datetime import datetime  
os.makedirs(f"outputs/detections/plates", exist_ok=True)
def load_model(model_path):
    """
    Load the pre-trained YOLO model for object detection.
    
    Args:
        model_path (str): Path to the YOLO model weights
    
    Returns:
        model: Loaded YOLO detection model
    """
    return YOLO(model_path)

def preprocess_frame(frame, input_size=(640, 640)):
    """
    Preprocess input frame for model inference.
    
    Args:
        frame (numpy.ndarray): Input image frame
        input_size (tuple): Target size for model input
    
    Returns:
        numpy.ndarray: Preprocessed frame ready for model inference
    """
    return cv2.resize(frame, input_size)

def detect_objects(model, frame):
    """
    Detect objects in a given frame using YOLO model.
    
    Args:
        model: Loaded YOLO detection model
        frame (numpy.ndarray): Input image frame
    
    Returns:
        list: Detected object bounding boxes
    """
    return model(frame)[0]

def detect_number_plates(model, frame):
    """
    Detect number plates in a given frame using YOLO model.
    
    Args:
        model: Loaded YOLO detection model
        frame (numpy.ndarray): Input image frame
    
    Returns:
        list: Detected number plate bounding boxes
    """
    return model(frame)[0]

def recognize_plate_text(plate_image):
    """
    Recognize text from a number plate image using OCR.
    
    Args:
        plate_image (numpy.ndarray): Cropped number plate image
    
    Returns:
        str: Recognized number plate text
    """
    # Try multiple thresholding methods
    _, plate_image_thresh1 = cv2.threshold(plate_image, 64, 255, cv2.THRESH_BINARY_INV)
    _, plate_image_thresh2 = cv2.threshold(plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_image_adaptive = cv2.adaptiveThreshold(plate_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Try reading license plate with different thresholded images
    license_plate_text1, license_plate_text_score1 = read_license_plate(plate_image_thresh1)
    license_plate_text2, license_plate_text_score2 = read_license_plate(plate_image_thresh2)
    license_plate_text3, license_plate_text_score3 = read_license_plate(plate_image_adaptive)

    # Choose the best result
    if license_plate_text1 is not None:
        return license_plate_text1, license_plate_text_score1
    elif license_plate_text2 is not None:
        return license_plate_text2, license_plate_text_score2
    elif license_plate_text3 is not None:
        return license_plate_text3, license_plate_text_score3
    else:
        return None, None

def process_video(video_path, output_path):
    """
    Process video for object detection and number plate recognition.
    
    Args:
        video_path (str): Path to input video file
        output_path (str): Path to save processed video
    
    Returns:
        None: Saves processed video with object and number plate annotations
    """
    # Load models
    coco_model = load_model("yolov8n.pt")
    license_plate_detector = load_model(r"models\license_plate_detector.pt")

    # Load video
    cap = cv2.VideoCapture(video_path)
    # os.makedirs("detections", exist_ok=True)
    vehicles = [2, 3, 5, 7]

    # Initialize MOT tracker
    mot_tracker = Sort()

    # Initialize results dictionary
    results = {}

    # Read frames
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # Detect objects
            detections = detect_objects(coco_model, frame)
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                # If the detected object is a vehicle
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles
            detections_array = (
                np.asarray(detections_) if len(detections_) > 0 else np.empty((0, 5))
            )
            track_ids = mot_tracker.update(detections_array)

            # Detect number plates
            license_plates = detect_number_plates(license_plate_detector, frame)
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]
                    # Recognize license plate text
                    license_plate_text, license_plate_text_score = recognize_plate_text(cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY))
                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {
                            "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                            "license_plate": {
                                "bbox": [x1, y1, x2, y2],
                                "text": license_plate_text,
                                "bbox_score": score,
                                "text_score": license_plate_text_score,
                            },
                        }

    # Write results
    write_csv(results, output_path)

def main():
    """
    Main function to orchestrate object detection and number plate recognition process.
    Handles model loading, video processing, and result visualization.
    """
    video_path = "./videos/sample_trimmed.mp4"
    output_path = f"./outputs/tests_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    process_video(video_path, output_path)

if __name__ == "__main__":
    # Entry point of the script
    # Runs the main object detection and number plate recognition process
    main()