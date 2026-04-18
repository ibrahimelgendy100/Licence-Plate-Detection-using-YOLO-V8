from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from plate_reader import detect_and_read_plate
from api_client import forward_plate_data
import os

app = FastAPI(title="YOLOv8 Plate Detection Microservice")

@app.post("/upload")
async def receive_camera_image(image: UploadFile = File(...)):
    """
    Endpoint that the camera will hit to send its image.
    """
    try:
        image_bytes = await image.read()
        
        # 1. Process image
        plate_text, orig_img, cropped_img, annotated_img = detect_and_read_plate(image_bytes)
        
        # 2. Forward results if needed (Uncomment in production)
        # forward_result = forward_plate_data(plate_text, orig_img, cropped_img, annotated_img)
        
        return JSONResponse(status_code=200, content={
            "success": True,
            "plate_text": plate_text,
            "message": "Image processed successfully."
            # "forward_response": forward_result
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
