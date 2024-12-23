import ast
import cv2
import numpy as np
import pandas as pd
import csv
import logging
from ..config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.Logging.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=config.Logging.LOG_FILE
)

results_to_be_visualized=config.Paths.INTERPOLATED_CSV
output_path=config.Paths.OUTPUT_VIDEO

def draw_border(
    img,
    top_left,
    bottom_right,
    color=(0, 255, 0),
    thickness=10,
    line_length_x=200,
    line_length_y=200,
):
    """
    Draw a border around a bounding box on an image.

    Args:
        img (numpy.ndarray): Input image
        top_left (tuple): Top-left coordinates of the bounding box
        bottom_right (tuple): Bottom-right coordinates of the bounding box
        color (tuple, optional): Color of the border. Defaults to (0, 255, 0).
        thickness (int, optional): Thickness of the border. Defaults to 10.
        line_length_x (int, optional): Length of the border lines in the x-direction. Defaults to 200.
        line_length_y (int, optional): Length of the border lines in the y-direction. Defaults to 200.

    Returns:
        numpy.ndarray: Image with the border drawn
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(
        img, (x1, y2), (x1, y2 - line_length_y), color, thickness
    )  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(
        img, (x2, y2), (x2, y2 - line_length_y), color, thickness
    )  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def load_video(video_path):
    """
    Load a video from a file path.

    Args:
        video_path (str): Path to the video file

    Returns:
        cv2.VideoCapture: Video capture object
    """
    cap = cv2.VideoCapture(video_path)
    return cap


def load_results(results_csv):
    """
    Load detection results from a CSV file.

    Args:
        results_csv (str): Path to the CSV file containing detection results

    Returns:
        pandas.DataFrame: Detection results
    """
    results = pd.read_csv(results_csv)
    print("Columns in results DataFrame:", list(results.columns))
    return results


def extract_license_plate_info(results, car_id):
    """
    Extract license plate information for a specific car ID.

    Args:
        results (pandas.DataFrame): Detection results
        car_id (int): Car ID

    Returns:
        dict: License plate information
    """
    max_score = np.amax(results[results["car_id"] == car_id]["license_number_score"])
    license_plate_info = results[
        (results["car_id"] == car_id) & (results["license_number_score"] == max_score)
    ]["license_number"].iloc[0]
    return license_plate_info


def create_license_plate_dict(results):
    """
    Create a dictionary of license plate information for each car.

    Args:
        results (pandas.DataFrame): Detection results DataFrame

    Returns:
        dict: Dictionary with car_id as keys and license plate information as values
    """
    license_plate_dict = {}
    
    for car_id in results['car_id'].unique():
        # Filter results for this car_id
        car_results = results[results['car_id'] == car_id]
        
        # Find the best license plate based on text score
        if 'license_plate_text_score' in results.columns:
            best_idx = car_results['license_plate_text_score'].argmax()
            best_result = car_results.iloc[best_idx]
        else:
            best_result = car_results.iloc[0]
        
        # Attempt to get license plate information
        try:
            # Use license_plate_text instead of license_number
            license_plate_number = best_result['license_plate_text']
            
            # Construct bounding box coordinates
            x1 = best_result['license_plate_bbox_x1']
            y1 = best_result['license_plate_bbox_y1']
            x2 = best_result['license_plate_bbox_x2']
            y2 = best_result['license_plate_bbox_y2']
            
            # Create a placeholder license crop (you might want to modify this)
            license_crop = np.zeros((int(y2-y1), int(x2-x1), 3), dtype=np.uint8)
            
            license_plate_dict[int(car_id)] = {
                'license_plate_number': license_plate_number,
                'license_crop': license_crop
            }
        except Exception as e:
            print(f"Error processing car_id {car_id}: {e}")
            # Provide a default entry to prevent KeyError
            license_plate_dict[int(car_id)] = {
                'license_plate_number': '',
                'license_crop': np.zeros((100, 200, 3), dtype=np.uint8)
            }
    
    return license_plate_dict


def annotate_frame(frame, results, license_plate_dict, frame_nmr):
    """
    Annotate a frame with license plate information.

    Args:
        frame (numpy.ndarray): Input frame
        results (pandas.DataFrame): Detection results
        license_plate_dict (dict): Dictionary of license plate information
        frame_nmr (int): Frame number

    Returns:
        numpy.ndarray: Annotated frame
    """
    # Convert frame_nmr to a string and handle float/int conversion
    frame_nmr_str = str(float(frame_nmr))
    
    # Convert frame number column to string to match
    results['frame_nmr'] = results['frame_number'].astype(str)
    
    df_ = results[results["frame_nmr"] == frame_nmr_str]
    for row_indx in range(len(df_)):
        # draw car
        car_x1, car_y1, car_x2, car_y2 = (
            float(df_.iloc[row_indx]["car_bbox_x1"]),
            float(df_.iloc[row_indx]["car_bbox_y1"]),
            float(df_.iloc[row_indx]["car_bbox_x2"]),
            float(df_.iloc[row_indx]["car_bbox_y2"])
        )
        draw_border(
            frame,
            (int(car_x1), int(car_y1)),
            (int(car_x2), int(car_y2)),
            (0, 255, 0),
            25,
            line_length_x=200,
            line_length_y=200,
        )

        # draw license plate
        x1, y1, x2, y2 = (
            float(df_.iloc[row_indx]["license_plate_bbox_x1"]),
            float(df_.iloc[row_indx]["license_plate_bbox_y1"]),
            float(df_.iloc[row_indx]["license_plate_bbox_x2"]),
            float(df_.iloc[row_indx]["license_plate_bbox_y2"])
        )
        cv2.rectangle(
            frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12
        )

        # Safely get car_id and handle potential type issues
        car_id = int(df_.iloc[row_indx]["car_id"])

        # Safely get license plate information
        try:
            # Ensure license plate text is converted to string
            license_plate_text = str(license_plate_dict.get(car_id, {}).get('license_plate_number', ''))
            
            # Safely get license crop
            license_crop = license_plate_dict.get(car_id, {}).get('license_crop', 
                np.zeros((100, 200, 3), dtype=np.uint8))

            # Crop license plate (if dimensions are valid)
            H, W, _ = license_crop.shape
            if H > 0 and W > 0:
                frame[
                    int(car_y1) - H - 100 : int(car_y1) - 100,
                    int((car_x2 + car_x1 - W) / 2) : int((car_x2 + car_x1 + W) / 2),
                    :,
                ] = license_crop

                frame[
                    int(car_y1) - H - 400 : int(car_y1) - H - 100,
                    int((car_x2 + car_x1 - W) / 2) : int((car_x2 + car_x1 + W) / 2),
                    :,
                ] = (255, 255, 255)

            # Add license plate text
            if license_plate_text:
                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17,
                )

                # Extend horizontal dimensions by adding more padding
                horizontal_padding = 200  # Increased from previous padding
                vertical_padding = 50     # Added some vertical padding

                cv2.rectangle(
                    frame,
                    (
                        int((car_x2 + car_x1 - text_width) / 2 - horizontal_padding/2), 
                        int(car_y1 - H - 250 - vertical_padding)
                    ),
                    (
                        int((car_x2 + car_x1 + text_width) / 2 + horizontal_padding/2), 
                        int(car_y1 - H - 250 + (text_height / 2) + vertical_padding)
                    ),
                    (255, 255, 255),  # White background
                    -1  # Filled rectangle
                )

                cv2.putText(
                    frame,
                    license_plate_text,
                    (
                        int((car_x2 + car_x1 - text_width) / 2),
                        int(car_y1 - H - 250 + (text_height / 2)),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    (0, 0, 0),
                    17,
                )

        except Exception as e:
            print(f"Error in annotation: {e}")
            pass

    return frame


def main():
    """
    Main function to demonstrate license plate annotation.
    """
    try:
        # Use paths from config
        video_path = config.Paths.DEFAULT_VIDEO
        results_path = config.Paths.INTERPOLATED_CSV
        output_path = config.Paths.OUTPUT_VIDEO

        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Load detection results
        results = load_results(results_path)

        # Create license plate dictionary
        license_plate_dict = create_license_plate_dict(results)

        frame_nmr = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_nmr += 1
            annotated_frame = annotate_frame(frame, results, license_plate_dict, frame_nmr)
            out.write(annotated_frame)

        logging.info(f"Annotated video saved to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        cap.release()
        out.release()


if __name__ == "__main__":
    main()
