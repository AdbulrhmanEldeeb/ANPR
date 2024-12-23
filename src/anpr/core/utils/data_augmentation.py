import csv
import numpy as np
from scipy.interpolate import interp1d
results_path=r"outputs\tests_2024-12-23_19-54-53.csv"

def interpolate_bounding_boxes(data):
    """
    Interpolate missing bounding boxes for cars in a video sequence.

    Args:
        data (list): List of dictionaries containing frame number, car ID, car bounding box, and license plate bounding box.

    Returns:
        list: List of dictionaries containing interpolated bounding boxes for cars.
    """
    # Print input data for debugging
    print("Input data sample:")
    for row in data[:3]:
        print(row)

    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row["frame_number"]) for row in data])
    car_ids = np.array([int(float(row["car_id"])) for row in data])
    car_bboxes = np.array([
        [float(row["car_bbox_x1"]), float(row["car_bbox_y1"]), 
         float(row["car_bbox_x2"]), float(row["car_bbox_y2"])] 
        for row in data
    ])
    license_plate_bboxes = np.array([
        [float(row["license_plate_bbox_x1"]), float(row["license_plate_bbox_y1"]), 
         float(row["license_plate_bbox_x2"]), float(row["license_plate_bbox_y2"])] 
        for row in data
    ])

    # Add columns for license plate details if they don't exist
    license_plate_texts = []
    license_plate_bbox_scores = []
    license_plate_text_scores = []

    for row in data:
        license_plate_texts.append(row.get('license_plate_text', ''))
        license_plate_bbox_scores.append(row.get('license_plate_bbox_score', ''))
        license_plate_text_scores.append(row.get('license_plate_text_score', ''))

    license_plate_texts = np.array(license_plate_texts)
    license_plate_bbox_scores = np.array(license_plate_bbox_scores)
    license_plate_text_scores = np.array(license_plate_text_scores)

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:
        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []
        license_plate_texts_interpolated = []
        license_plate_bbox_scores_interpolated = []
        license_plate_text_scores_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]
            license_plate_text = license_plate_texts[car_mask][i]
            license_plate_bbox_score = license_plate_bbox_scores[car_mask][i]
            license_plate_text_score = license_plate_text_scores[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i - 1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]
                prev_license_plate_text = license_plate_texts_interpolated[-1]
                prev_license_plate_bbox_score = license_plate_bbox_scores_interpolated[-1]
                prev_license_plate_text_score = license_plate_text_scores_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number + 1, frame_number - 1, frames_gap - 1)

                    for j in range(frames_gap - 1):
                        interpolation_factor = (x_new[j] - prev_frame_number) / (frame_number - prev_frame_number)
                        interpolated_car_bbox = prev_car_bbox * (1 - interpolation_factor) + car_bbox * interpolation_factor
                        interpolated_license_plate_bbox = prev_license_plate_bbox * (1 - interpolation_factor) + license_plate_bbox * interpolation_factor

                        row = {
                            "frame_number": str(x_new[j]),
                            "car_id": str(car_id),
                            "car_bbox_x1": str(interpolated_car_bbox[0]),
                            "car_bbox_y1": str(interpolated_car_bbox[1]),
                            "car_bbox_x2": str(interpolated_car_bbox[2]),
                            "car_bbox_y2": str(interpolated_car_bbox[3]),
                            "license_plate_bbox_x1": str(interpolated_license_plate_bbox[0]),
                            "license_plate_bbox_y1": str(interpolated_license_plate_bbox[1]),
                            "license_plate_bbox_x2": str(interpolated_license_plate_bbox[2]),
                            "license_plate_bbox_y2": str(interpolated_license_plate_bbox[3]),
                            "license_plate_bbox_score": str(prev_license_plate_bbox_score),
                            "license_plate_text": str(prev_license_plate_text),
                            "license_plate_text_score": str(prev_license_plate_text_score)
                        }
                        interpolated_data.append(row)

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)
            license_plate_texts_interpolated.append(license_plate_text)
            license_plate_bbox_scores_interpolated.append(license_plate_bbox_score)
            license_plate_text_scores_interpolated.append(license_plate_text_score)

            row = {
                "frame_number": str(frame_number),
                "car_id": str(car_id),
                "car_bbox_x1": str(car_bbox[0]),
                "car_bbox_y1": str(car_bbox[1]),
                "car_bbox_x2": str(car_bbox[2]),
                "car_bbox_y2": str(car_bbox[3]),
                "license_plate_bbox_x1": str(license_plate_bbox[0]),
                "license_plate_bbox_y1": str(license_plate_bbox[1]),
                "license_plate_bbox_x2": str(license_plate_bbox[2]),
                "license_plate_bbox_y2": str(license_plate_bbox[3]),
                "license_plate_bbox_score": str(license_plate_bbox_score),
                "license_plate_text": str(license_plate_text),
                "license_plate_text_score": str(license_plate_text_score)
            }
            interpolated_data.append(row)

    return interpolated_data

# Load the CSV file
with open(results_path, "r") as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = [
    "frame_number", "car_id", 
    "car_bbox_x1", "car_bbox_y1", "car_bbox_x2", "car_bbox_y2", 
    "license_plate_bbox_x1", "license_plate_bbox_y1", "license_plate_bbox_x2", "license_plate_bbox_y2", 
    "license_plate_bbox_score", "license_plate_text", "license_plate_text_score"
]
with open(results_path.replace(".csv", "_interpolated.csv"), "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)
