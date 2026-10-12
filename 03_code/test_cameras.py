"""Capture a frame from each camera and save as image to identify LookCam."""
import cv2

for i in range(2):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            filename = f"03_code/camera_{i}_test.jpg"
            cv2.imwrite(filename, frame)
            h, w = frame.shape[:2]
            # Check if frame is all black
            avg = frame.mean()
            print(f"Camera {i}: {w}x{h}, avg brightness={avg:.1f}, saved to {filename}")
            if avg < 5:
                print(f"  >> Camera {i} is BLACK (not a real feed)")
            else:
                print(f"  >> Camera {i} has a REAL image!")
        else:
            print(f"Camera {i}: opened but cannot read frame")
        cap.release()
    else:
        print(f"Camera {i}: not available")
