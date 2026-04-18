import cv2
import easyocr
import numpy as np
import os
from ultralytics import YOLO
import pathlib

# Fix for pathlib issue on Windows sometimes when loading linux-trained models
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best.pt")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model from {MODEL_PATH}: {e}")
    model = None

# Initialize EasyOCR reader
try:
    reader = easyocr.Reader(['en'], gpu=True) # or False if no GPU
except Exception as e:
    print(f"Error loading EasyOCR: {e}")
    reader = None

def detect_and_read_plate(image_bytes: bytes):
    """
    Takes image bytes, returns a tuple:
    (plate_text, original_image, cropped_plate_image, annotated_image)
    Images are returned as bytes (JPEG encoded).
    """
    if model is None or reader is None:
        raise RuntimeError("Model or OCR reader not initialized properly.")

    # Convert bytes to numpy array, then to OpenCV image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image bytes.")

    original_img = img.copy()
    
    # Run YOLOv8 inference
    results = model(img)
    
    plate_text = ""
    cropped_plate_bytes = None
    annotated_bytes = None
    
    # Find the best detection
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            continue
            
        # Get the first box (assuming one plate per image for simplicity, or highest conf)
        box = boxes[0] 
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Validate crop boundaries
        y1, y2 = max(0, y1), min(original_img.shape[0], y2)
        x1, x2 = max(0, x1), min(original_img.shape[1], x2)
        
        # Crop the plate
        cropped_plate = original_img[y1:y2, x1:x2]
        
        if cropped_plate.size == 0:
             continue
        
        # Use EasyOCR to read the text
        ocr_result = reader.readtext(cropped_plate)
        if ocr_result:
            # Join all text pieces recognized
            plate_text = " ".join([res[1] for res in ocr_result])
        
        # Draw bounding box on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, plate_text, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Encode cropped plate
        _, cropped_buf = cv2.imencode('.jpg', cropped_plate)
        cropped_plate_bytes = cropped_buf.tobytes()
        break
        
    # Encode annotated image
    _, annotated_buf = cv2.imencode('.jpg', img)
    annotated_bytes = annotated_buf.tobytes()
    
    return plate_text, image_bytes, cropped_plate_bytes, annotated_bytes
