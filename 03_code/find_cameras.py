"""List all available camera devices to find LookCam's index."""
import cv2

print("Scanning camera devices...\n")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  [OK] Camera {i}: {w}x{h}  (frame read: {ret})")
        cap.release()
    else:
        print(f"  [--] Camera {i}: not available")

print("\n>> Use the camera index that corresponds to LookCam.")
print("   (Usually index 1 or 2 if your laptop webcam is index 0)")
