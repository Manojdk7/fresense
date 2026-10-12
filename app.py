from flask import Flask, Response, render_template
import cv2
import numpy as np
import mss
import pygetwindow as gw
import time

app = Flask(__name__)

sct = mss.mss()


def get_lookcam_window():
    windows = gw.getAllWindows()

    for window in windows:
        title = window.title.lower()

        if "lookcam" in title and "chrome" not in title and "edge" not in title:
            return window

    return None


def generate_frames():

    while True:

        window = get_lookcam_window()

        if window is None:
            print("LookCam window not found...")
            time.sleep(1)
            continue

        # Get LookCam window position
        left = window.left
        top = window.top
        width = window.width
        height = window.height

        if width <= 0 or height <= 0:
            continue

        monitor = {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        }

        # Capture LookCam window area
        screenshot = sct.grab(monitor)

        # Convert to OpenCV image
        frame = np.array(screenshot)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # --- CROP THE VIDEO ---
        # The LookCam window has a top toolbar and a right-side panel
        # We crop them out here.
        y1 = 75           # Top edge (toolbar)
        y2 = height - 10  # Bottom edge
        x1 = 5            # Left edge
        x2 = width - 150  # Right edge

        # Ensure coordinates are within bounds
        if y2 > y1 and x2 > x1:
            frame = frame[y1:y2, x1:x2]

        # -----------------------------
        # YOUR FOODSAFE AI PROCESSING
        # -----------------------------
        # Example:
        # result = model.predict(frame)

        # Encode frame as JPEG
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        # Send frame to browser
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <!DOCTYPE html>

    <html>

    <head>
        <title>FoodSafe IoT</title>

        <style>

            body {
                background: #111;
                color: white;
                font-family: Arial;
                text-align: center;
            }

            h1 {
                margin-top: 30px;
            }

            img {
                width: 80%;
                max-width: 900px;
                border: 3px solid white;
                border-radius: 10px;
            }

        </style>

    </head>

    <body>

        <h1>FoodSafe IoT</h1>

        <h2>Live Camera Feed</h2>

        <img src="/video_feed">

    </body>

    </html>
    """


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True
    )
