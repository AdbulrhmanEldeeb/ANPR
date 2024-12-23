# Automatic Number Plate Recognition (ANPR)

## Project Overview
Real-time Automatic Number Plate Recognition system using computer vision and machine learning techniques.

## Features
- Real-time license plate detection
- Optical Character Recognition (OCR)
- Multi-vehicle tracking
- High accuracy performance


### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup
1. Clone the repository
```bash
git clone https://github.com/AdbulrhmanEldeeb/ANPR.git
cd ANPR
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt 
```

## Usage
1. run main.py for detection and recognition you will  get the results in outputs folder as csv file ,specify the video path and output path in main.py .
```bash
python src/anpr/core/main.py
```
2. run data_augmentation.py for data augmentation and processing missing frames  , specify the results.csv file path in data_augmentation.py and interpolated file path 

```bash
python src/anpr/core/data_augmentation.py
```
3. run visualize.py for visualizing the results , specify the interploated_results.csv file path in  and video path and output video path in visualize.py
```bash
python src/anpr/core/visualize.py
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License

## Acknowledgements
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- EasyOCR

## download video for test 
https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/
