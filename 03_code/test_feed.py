"""Test if /video_feed endpoint returns MJPEG data."""
import urllib.request
try:
    r = urllib.request.urlopen('http://127.0.0.1:5000/video_feed', timeout=8)
    data = r.read(4096)
    ct = r.headers.get('Content-Type')
    print(f"Got {len(data)} bytes")
    print(f"Content-Type: {ct}")
    if b'--frame' in data:
        print("SUCCESS: MJPEG stream is working!")
    elif b'\xff\xd8' in data:
        print("SUCCESS: JPEG data detected in stream!")
    else:
        print("WARNING: Data received but no JPEG markers found")
    print(f"First 100 bytes: {data[:100]}")
except Exception as e:
    print(f"ERROR: {e}")
