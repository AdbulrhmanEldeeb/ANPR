import os
from datetime import datetime

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
    
    # Timestamp for unique output naming
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Input and Output Paths
    class Paths:
        # Video Inputs
        VIDEO_DIR = os.path.join(PROJECT_ROOT, 'videos')
        DEFAULT_VIDEO = os.path.join(VIDEO_DIR, 'sample_trimmed.mp4')
        
        # Output Directories
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
        CSV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'csv')
        VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'videos')
        
        # Specific Output Files
        RESULTS_CSV = os.path.join(CSV_OUTPUT_DIR, f'results_{TIMESTAMP}.csv')
        INTERPOLATED_CSV = os.path.join(CSV_OUTPUT_DIR, f'interpolated_{TIMESTAMP}.csv')
        OUTPUT_VIDEO = os.path.join(VIDEO_OUTPUT_DIR, f'annotated_{TIMESTAMP}.mp4')
    
    # Detection Configuration
    class Detection:
        CONFIDENCE_THRESHOLD = 0.5
        NMS_THRESHOLD = 0.4
    
    # Interpolation Configuration
    class Interpolation:
        FRAME_GAP_THRESHOLD = 1
    
    # Visualization Configuration
    class Visualization:
        BORDER_COLOR = (0, 255, 0)  # Green
        TEXT_COLOR = (0, 0, 0)      # Black
        BACKGROUND_COLOR = (255, 255, 255)  # White
    
    # Logging Configuration
    class Logging:
        LOG_LEVEL = 'INFO'
        LOG_FILE = os.path.join(OUTPUT_DIR, f'log_{TIMESTAMP}.log')

# Ensure output directories exist
os.makedirs(Config.Paths.CSV_OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.Paths.VIDEO_OUTPUT_DIR, exist_ok=True)

# Export configuration for easy import
config = Config()
