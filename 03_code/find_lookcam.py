"""Test LookCam stream URLs with OpenCV."""
import cv2

ip = "10.77.123.111"
port = 8000

urls = [
    f"http://{ip}:{port}/video",
    f"http://{ip}:{port}/videofeed",
    f"http://{ip}:{port}/stream",
    f"http://{ip}:{port}/mjpeg",
    f"http://{ip}:{port}/mjpegfeed",
    f"http://{ip}:{port}/shot.jpg",
    f"http://{ip}:{port}/",
    f"http://{ip}:{port}/cam",
    f"http://{ip}:{port}/live",
    f"http://{ip}:{port}/api/stream",
    f"http://{ip}:{port}/videostream",
    f"http://{ip}:{port}/capture",
]

for url in urls:
    print(f"Trying: {url} ... ", end="", flush=True)
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            avg = frame.mean()
            h, w = frame.shape[:2]
            print(f"SUCCESS! {w}x{h}, brightness={avg:.1f}")
            cv2.imwrite("03_code/lookcam_stream.jpg", frame)
            cap.release()
            print(f"\n*** WORKING URL: {url} ***")
            print(f"Frame saved to 03_code/lookcam_stream.jpg")
            break
        else:
            print("opened but no frame")
        cap.release()
    else:
        print("failed")
else:
    print("\nNo URL worked. Try checking LookCam app for the stream URL.")
