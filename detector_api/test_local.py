import os
import sys

# Change working directory so models load correctly if relying on relative paths internally
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from plate_reader import detect_and_read_plate

demo_path = os.path.join("..", "demo2.jpeg")

if not os.path.exists(demo_path):
    print(f"Could not find demo file at {demo_path}")
    sys.exit(1)

with open(demo_path, "rb") as f:
    img_bytes = f.read()

try:
    print("Running detection...")
    plate_text, orig, cropped, annotated = detect_and_read_plate(img_bytes)
    print(f"Success! Detected Plate Text: '{plate_text}'")
    print(f"Original Size: {len(orig)} bytes")
    if cropped:
        print(f"Cropped Size: {len(cropped)} bytes")
    if annotated:
        print(f"Annotated Size: {len(annotated)} bytes")
except Exception as e:
    print(f"Test failed: {e}")
