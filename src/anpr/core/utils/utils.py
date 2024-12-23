import string
import easyocr
import cv2
import numpy as np
import csv

# Initialize the OCR reader
reader = easyocr.Reader(["en"], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}

dict_int_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.

    Returns:
        None
    """
    # Prepare CSV headers
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_number', 'car_id', 
            'car_bbox_x1', 'car_bbox_y1', 'car_bbox_x2', 'car_bbox_y2', 
            'license_plate_bbox_x1', 'license_plate_bbox_y1', 
            'license_plate_bbox_x2', 'license_plate_bbox_y2', 
            'license_plate_text', 'license_plate_bbox_score', 
            'license_plate_text_score'
        ])
        
        # Write results for each frame
        for frame_number, frame_data in results.items():
            for car_id, car_info in frame_data.items():
                writer.writerow([
                    frame_number, car_id,
                    car_info['car']['bbox'][0], car_info['car']['bbox'][1],
                    car_info['car']['bbox'][2], car_info['car']['bbox'][3],
                    car_info['license_plate']['bbox'][0], car_info['license_plate']['bbox'][1],
                    car_info['license_plate']['bbox'][2], car_info['license_plate']['bbox'][3],
                    car_info['license_plate']['text'], 
                    car_info['license_plate']['bbox_score'], 
                    car_info['license_plate']['text_score']
                ])


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    # Check if the text length is 7 characters
    if len(text) != 7:
        return False

    # Check if the text matches the required format
    if (
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys())
        and (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys())
        and (
            text[2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[2] in dict_char_to_int.keys()
        )
        and (
            text[3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[3] in dict_char_to_int.keys()
        )
        and (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys())
        and (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys())
        and (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys())
    ):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    # Initialize the formatted license plate text
    license_plate_ = ""
    
    # Define the mapping dictionaries for each character position
    mapping = {
        0: dict_int_to_char,
        1: dict_int_to_char,
        4: dict_int_to_char,
        5: dict_int_to_char,
        6: dict_int_to_char,
        2: dict_char_to_int,
        3: dict_char_to_int,
    }
    
    # Iterate over each character in the text
    for j in [0, 1, 2, 3, 4, 5, 6]:
        # Check if the character needs to be converted
        if text[j] in mapping[j].keys():
            # Convert the character using the mapping dictionary
            license_plate_ += mapping[j][text[j]]
        else:
            # Keep the character as is
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read and recognize license plate text using OCR.

    Args:
        license_plate_crop (numpy.ndarray): Preprocessed license plate image

    Returns:
        tuple: Recognized license plate text and confidence score
    """
    # Save the cropped image for debugging
    cv2.imwrite(
        f"outputs/detections/plates/debug_license_plate_{np.random.randint(1000)}.jpg", license_plate_crop
    )

    # Perform OCR on license plate
    detections = reader.readtext(license_plate_crop)
    print("EasyOCR Detections:", detections)

    # Process OCR results
    for detection in detections:
        bbox, text, score = detection

        # Preprocess the text
        text = text.upper().replace(" ", "")
        print(f"Detected text: {text}, Score: {score}")

        # Check if the text complies with the required format
        if license_complies_format(text):
            # Format the license plate text
            formatted_text = format_license(text)
            return formatted_text, score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Match license plate to the corresponding vehicle tracking ID.

    Args:
        license_plate (tuple): License plate bounding box coordinates
        vehicle_track_ids (list): Vehicle tracking IDs and bounding boxes

    Returns:
        tuple: Vehicle bounding box coordinates and tracking ID
    """
    # Extract license plate coordinates
    x1, y1, x2, y2, score, class_id = license_plate
    
    # Calculate license plate center
    license_plate_center_x = np.mean([x1, x2])
    license_plate_center_y = np.mean([y1, y2])
    
    # Find the best matching vehicle
    for track_id in vehicle_track_ids:
        car_x1, car_y1, car_x2, car_y2, car_id = track_id
        
        # Calculate vehicle center
        car_center_x = np.mean([car_x1, car_x2])
        car_center_y = np.mean([car_y1, car_y2])
        
        # Check if license plate is within vehicle bounding box
        if (car_x1 <= license_plate_center_x <= car_x2 and 
            car_y1 <= license_plate_center_y <= car_y2):
            return car_x1, car_y1, car_x2, car_y2, car_id
    
    return -1, -1, -1, -1, -1
