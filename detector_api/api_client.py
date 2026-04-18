import requests

FORWARD_URL = "http://example.com/api/receive_plate" # To be replaced by the user with the actual destination URL

def forward_plate_data(plate_text: str, original_bytes: bytes, cropped_bytes: bytes, annotated_bytes: bytes):
    """
    Sends the plate text and the three images to the destination backend.
    """
    data = {
        "license_plate": plate_text
    }
    
    files = {
        "original_image": ("original.jpg", original_bytes, "image/jpeg"),
    }
    
    if cropped_bytes:
        files["cropped_image"] = ("cropped.jpg", cropped_bytes, "image/jpeg")
        
    if annotated_bytes:
        files["annotated_image"] = ("annotated.jpg", annotated_bytes, "image/jpeg")
        
    try:
        response = requests.post(FORWARD_URL, data=data, files=files, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error forwarding plate data: {e}")
        return {"error": str(e)}
