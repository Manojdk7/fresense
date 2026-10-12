from flask import Flask, request, jsonify, send_from_directory
from flask import send_file
from flask_cors import CORS
from risk_engine import RiskEngine
from database import db, init_database
from models import Device, DangerZoneLog, RiskAnalysis, SensorReading
from datetime import datetime, timezone
from pathlib import Path
import os
import io
import time
import json
import tempfile
import uuid
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_IMAGES_DIR = BASE_DIR / 'result_images'
RESULT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(BASE_DIR / '04_web'))
CORS(app)
init_database(app)

engine = None

# In-memory storage for latest sensor data from ESP32 devices
sensor_data_store = {}

# ESP32 SEND_INTERVAL_MS is 5000 — allow a few missed packets before offline
SENSOR_TIMEOUT_SECONDS = 15

# Latest AI image-analysis alarm (source of truth: RiskEngine is_safe)
ai_alarm_state = {
    'ai_high_risk': False,
    'top_prediction': None,
    'final_score': None,
    'is_safe': None,
    'timestamp': None,
    'alarm_id': 0,
    'alarm_until': 0.0,
}


def _update_ai_alarm_from_result(result):
    """Set physical AI alarm from RiskEngine image analysis (is_safe == False → HIGH RISK)."""
    global ai_alarm_state
    is_safe = bool(result.get('is_safe', True))
    now_ts = time.time()
    current_alarm_id = ai_alarm_state.get('alarm_id', 0)
    if not is_safe:
        current_alarm_id += 1
        # Active pulse window: 15s allows ESP32 (sending every 5s) to catch and trigger
        alarm_until = now_ts + 15.0
    else:
        alarm_until = 0.0

    ai_alarm_state = {
        'ai_high_risk': not is_safe,
        'top_prediction': result.get('top_prediction'),
        'final_score': result.get('final_score'),
        'is_safe': is_safe,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'alarm_id': current_alarm_id,
        'alarm_until': alarm_until,
    }
    print(
        f"[AI-ALARM] is_safe={is_safe} ai_high_risk={ai_alarm_state['ai_high_risk']} "
        f"alarm_id={current_alarm_id} pred={ai_alarm_state['top_prediction']} score={ai_alarm_state['final_score']}",
        flush=True,
    )


def _is_ai_alarm_active():
    """Return whether the AI high-risk alarm is currently active and within its pulse window."""
    if not ai_alarm_state.get('ai_high_risk', False):
        return False
    if time.time() > ai_alarm_state.get('alarm_until', 0.0):
        ai_alarm_state['ai_high_risk'] = False
        return False
    return True


def _utc_iso(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _sensor_reading_payload(reading):
    return {
        'device_id': reading.device_id,
        'source': reading.source,
        'gas': reading.gas,
        'temperature': reading.temperature,
        'humidity': reading.humidity,
        'mq135_raw': reading.mq135_raw,
        'mq2_digital': reading.mq2_digital,
        'alarm': reading.alarm,
        'timestamp': _utc_iso(reading.timestamp),
    }


def _store_sensor_reading(data, timestamp):
    reading = SensorReading(
        device_id=data['device_id'],
        source=data.get('source', 'legacy'),
        gas=data['gas'],
        temperature=data['temperature'],
        humidity=data['humidity'],
        mq135_raw=data.get('mq135_raw'),
        mq2_digital=data.get('mq2_digital'),
        alarm=data.get('alarm'),
        timestamp=timestamp,
    )
    device = Device.query.filter_by(device_id=data['device_id']).first()
    if device is None:
        device = Device(device_id=data['device_id'], name=data['device_id'])
        db.session.add(device)
    device.last_seen = timestamp
    device.status = 'connected'
    db.session.add(reading)
    db.session.commit()


def _log_danger_zone(device_id: str, temperature: float, timestamp):
    """Append a DangerZoneLog row; compute cumulative minutes in the danger zone.
    The bacterial danger zone is 5°C – 60°C. The ESP32 sends every ~5 seconds, so
    we credit 5 seconds (≈ 0.0833 min) per reading that falls inside the zone."""
    DANGER_LOW, DANGER_HIGH = 5.0, 60.0
    # Approximate interval per reading (match ESP32 SEND_INTERVAL_MS)
    INTERVAL_MINUTES = 5.0 / 60.0  # 5 seconds → minutes

    in_zone = DANGER_LOW <= temperature <= DANGER_HIGH
    # Fetch latest cumulative total for this device
    last_log = (
        DangerZoneLog.query
        .filter_by(device_id=device_id)
        .order_by(DangerZoneLog.timestamp.desc())
        .first()
    )
    prev_cumulative = last_log.cumulative_minutes if last_log else 0.0
    new_cumulative = prev_cumulative + (INTERVAL_MINUTES if in_zone else 0.0)

    log_entry = DangerZoneLog(
        device_id=device_id,
        temperature=temperature,
        in_danger_zone=in_zone,
        cumulative_minutes=round(new_cumulative, 4),
        timestamp=timestamp,
    )
    db.session.add(log_entry)
    db.session.commit()


def _store_risk_analysis(
    result,
    device_id=None,
    image_filename=None,
    image_mime_type=None,
    image_data=None,
    image_path=None,
):
    analysis = RiskAnalysis(
        device_id=device_id,
        image_filename=image_filename,
        image_mime_type=image_mime_type,
        image_data=image_data,
        image_path=image_path,
        image_prediction=result.get('top_prediction'),
        image_score=result.get('image_score'),
        sensor_score=result.get('sensor_score'),
        final_score=result.get('final_score'),
        is_safe=result.get('is_safe'),
        ai_high_risk=not bool(result.get('is_safe', True)),
        timestamp=datetime.now(timezone.utc),
    )
    db.session.add(analysis)
    db.session.commit()

import threading
import time as _cam_time
import numpy as np
import cv2
import requests
import re
import os

# ==========================================
# ESP32-CAM CONFIGURATION
# ==========================================
# Default ESP32 IP, user can change it via the dashboard
ESP32_IP = os.environ.get("ESP32_IP", "")

class ESP32CameraThread:
    """
    Robustly captures MJPEG stream from ESP32-CAM using requests to prevent OpenCV lag/buffer buildup.
    """
    def __init__(self):
        self.frame_bytes = None
        self.lock = threading.Lock()
        self.running = False
        self._thread = None
        self.last_error = "Not connected"
        self.connected = False

    def _run(self):
        global ESP32_IP
        
        while self.running:
            if not ESP32_IP:
                self.last_error = "ESP32 IP not configured"
                self.connected = False
                _cam_time.sleep(1.0)
                continue
                
            stream_url = f"http://{ESP32_IP}:81/stream"
            
            try:
                with requests.Session() as session:
                    res = session.get(stream_url, stream=True, timeout=(5, 5))
                    
                    if res.status_code != 200:
                        self.last_error = f"HTTP {res.status_code} on stream connect"
                        self.connected = False
                        _cam_time.sleep(2.0)
                        continue
                    
                    self.connected = True
                    self.last_error = None
                    bytes_data = bytes()
                    
                    for chunk in res.iter_content(chunk_size=4096):
                        if not self.running:
                            break
                            
                        bytes_data += chunk
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        
                        if a != -1 and b != -1:
                            if b > a:
                                jpg = bytes_data[a:b+2]
                                bytes_data = bytes_data[b+2:]
                                
                                # Decode to check validity (optional, but ensures stability)
                                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                if frame is not None:
                                    with self.lock:
                                        self.frame_bytes = jpg
                            else:
                                bytes_data = bytes_data[a:]
                                
            except requests.exceptions.RequestException as e:
                self.last_error = f"Stream error: {str(e)[:50]}"
                self.connected = False
                _cam_time.sleep(2.0)
            except Exception as e:
                self.last_error = f"Unexpected error: {str(e)[:50]}"
                self.connected = False
                _cam_time.sleep(2.0)

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_jpeg(self):
        with self.lock:
            return self.frame_bytes

    def status(self):
        with self.lock:
            has_frame = self.frame_bytes is not None
        return {
            'configured_ip': ESP32_IP,
            'connected': self.connected,
            'has_frame': has_frame,
            'running': self.running,
            'last_error': self.last_error,
        }

_camera_thread = None
_camera_lock = threading.Lock()

def get_camera_thread():
    global _camera_thread
    with _camera_lock:
        if _camera_thread is None or not _camera_thread.running:
            _camera_thread = ESP32CameraThread()
            _camera_thread.start()
            _cam_time.sleep(1.0) 
    return _camera_thread

def generate_esp32_frames():
    cam = get_camera_thread()
    while True:
        jpeg = cam.get_jpeg()
        if not jpeg:
            # Generate placeholder image if no stream
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "ESP32-CAM Offline", (150, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            if cam.last_error:
                cv2.putText(frame, str(cam.last_error), (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Enter IP in dashboard to connect", (110, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            
            success, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                jpeg = buf.tobytes()
                
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
        _cam_time.sleep(0.04)

from flask import Response, request, jsonify

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_esp32_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route('/api/camera/status')
def camera_status():
    cam = get_camera_thread()
    return jsonify(cam.status())

@app.route("/set_ip", methods=["POST"])
def set_ip():
    global ESP32_IP
    data = request.json
    if data and "ip" in data:
        ip_str = data["ip"].strip()
        if ip_str.startswith("http://"):
            ip_str = ip_str[7:]
        elif ip_str.startswith("https://"):
            ip_str = ip_str[8:]
        if ip_str.endswith("/"):
            ip_str = ip_str[:-1]
            
        ESP32_IP = ip_str
        get_camera_thread() 
        return jsonify({"status": "success", "ip": ESP32_IP})
    return jsonify({"status": "error", "message": "Invalid data"}), 400

@app.route("/control")
def control():
    global ESP32_IP
    if not ESP32_IP:
        return jsonify({"error": "ESP32 IP not set"}), 400
        
    var = request.args.get('var')
    val = request.args.get('val')
    
    if not var or not val:
        return jsonify({"error": "Missing var or val"}), 400
        
    url = f"http://{ESP32_IP}/control?var={var}&val={val}"
    try:
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": f"ESP32 returned HTTP {res.status_code}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/capture")
def capture():
    global ESP32_IP
    if not ESP32_IP:
        return jsonify({"error": "ESP32 IP not set"}), 400
        
    url = f"http://{ESP32_IP}/capture"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            from datetime import datetime
            save_dir = RESULT_IMAGES_DIR
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = save_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(res.content)
                
            return jsonify({"status": "success", "path": str(filepath), "message": f"Saved as {filename}"})
        else:
            return jsonify({"error": f"ESP32 returned HTTP {res.status_code}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/favicon.ico')
def favicon():
    """Silence browser favicon 404; optional file can be placed in 04_web/."""
    favicon_path = Path(app.static_folder) / 'favicon.ico'
    if favicon_path.exists():
        return send_from_directory(app.static_folder, 'favicon.ico')
    return ('', 204)

@app.route('/analyze', methods=['POST'])
def analyze():
    global engine
    if engine is None:
        engine = RiskEngine()
        
    image_data = None
    image_filename = None
    image_mime_type = None

    if 'image' in request.files and request.files['image'].filename:
        image = request.files['image']
        image_filename = image.filename or 'uploaded-image.jpg'
        image_mime_type = image.mimetype or 'application/octet-stream'
        image_data = image.read()
    elif request.form.get('use_lookcam') == 'true':
        jpeg = get_camera_thread().get_jpeg()
        if not jpeg:
            return jsonify({'error': 'ESP32 camera stream not available. Set IP Address.'}), 400
        image_data = jpeg
        image_filename = 'esp32-camera-capture.jpg'
        image_mime_type = 'image/jpeg'
    else:
        return jsonify({'error': 'No image uploaded and camera capture not requested'}), 400

    try:
        temp = float(request.form.get('temperature', 25.0))
        hum = float(request.form.get('humidity', 60.0))
        gas = float(request.form.get('gas', 100.0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Sensor values must be numeric'}), 400
    suffix = Path(image_filename).suffix or '.jpg'
    temp_img_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_img_path = temp_file.name
            temp_file.write(image_data)

        category = request.form.get('category', None)
        latest_reading = SensorReading.query.order_by(SensorReading.timestamp.desc()).first()
        result = engine.analyze(
            temp_img_path,
            temp,
            hum,
            gas,
            latest_reading.mq135_raw if latest_reading else None,
            latest_reading.mq2_digital if latest_reading else None,
            latest_reading.alarm if latest_reading else None,
            category=category,
        )
        _update_ai_alarm_from_result(result)
        try:
            device_id = request.form.get('device_id')
            if device_id is None:
                latest_reading = SensorReading.query.order_by(SensorReading.timestamp.desc()).first()
                device_id = latest_reading.device_id if latest_reading else None
            safe_name = secure_filename(image_filename) or 'uploaded-image.jpg'
            stored_filename = f'{uuid.uuid4().hex}_{safe_name}'
            stored_path = RESULT_IMAGES_DIR / stored_filename
            stored_path.write_bytes(image_data)
            relative_image_path = stored_path.relative_to(BASE_DIR).as_posix()
            _store_risk_analysis(
                result,
                device_id,
                image_filename,
                image_mime_type,
                image_data,
                relative_image_path,
            )
        except Exception as db_error:
            db.session.rollback()
            print(f"[DATABASE] Could not store risk analysis: {db_error}", flush=True)
    except ValueError as e:
        err_str = str(e)
        if "Non-food item detected" in err_str:
            return jsonify({
                'success': False,
                'error': err_str,
                'is_food': False,
                'error_type': 'OOD_REJECTION'
            }), 422
        return jsonify({'error': err_str}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
    return jsonify(result)

# ====================================================================
# Phase 4: Hardware Integration — Sensor API Endpoints
# ====================================================================

@app.route('/api/sensors/data', methods=['POST'])
def receive_sensor_data():
    """Receive sensor data from ESP32 device."""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400
        
        # Validate required fields
        required_fields = ['device_id', 'gas', 'temperature', 'humidity']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        # Validate numeric values
        try:
            gas = float(data['gas'])
            temperature = float(data['temperature'])
            humidity = float(data['humidity'])
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Sensor values must be numeric'}), 400
        
        # Basic range validation
        if not (0 <= gas <= 10000):
            return jsonify({'success': False, 'error': 'Gas value out of valid range (0-10000)'}), 400
        if not (-40 <= temperature <= 125):
            return jsonify({'success': False, 'error': 'Temperature out of valid range (-40 to 125)'}), 400
        if not (0 <= humidity <= 100):
            return jsonify({'success': False, 'error': 'Humidity out of valid range (0-100)'}), 400
        
        device_id = str(data['device_id'])
        source = str(data.get('source', 'legacy')).strip().lower()
        if source not in {'hardware', 'simulation', 'legacy'}:
            return jsonify({'success': False, 'error': 'Source must be hardware, simulation, or legacy'}), 400
        timestamp_dt = datetime.now(timezone.utc)
        timestamp = timestamp_dt.isoformat()

        # Optional ESP32 fields (preserve firmware payload names)
        mq135_raw = data.get('mq135_raw')
        mq2_digital = data.get('mq2_digital')
        alarm = data.get('alarm')

        try:
            mq135_raw = int(mq135_raw) if mq135_raw is not None else None
        except (ValueError, TypeError):
            mq135_raw = None

        try:
            mq2_digital = int(mq2_digital) if mq2_digital is not None else None
        except (ValueError, TypeError):
            mq2_digital = None

        if alarm is not None:
            alarm = bool(alarm)

        # Store latest reading (full ESP32 payload + server timestamp)
        sensor_data_store[device_id] = {
            'device_id': device_id,
            'source': source,
            'gas': gas,
            'temperature': temperature,
            'humidity': humidity,
            'mq135_raw': mq135_raw,
            'mq2_digital': mq2_digital,
            'alarm': alarm,
            'timestamp': timestamp
        }

        try:
            _store_sensor_reading(sensor_data_store[device_id], timestamp_dt)
            _log_danger_zone(device_id, temperature, timestamp_dt)
        except Exception as db_error:
            db.session.rollback()
            print(f"[DATABASE] Could not store sensor reading: {db_error}", flush=True)

        client_ip = request.remote_addr
        print(
            f"[SENSOR] from {client_ip} device={device_id}: "
            f"gas={gas}, temp={temperature}, hum={humidity}, "
            f"mq135_raw={mq135_raw}, mq2_digital={mq2_digital}, alarm={alarm}",
            flush=True
        )

        return jsonify({
            'success': True,
            'message': 'Sensor data received successfully',
            'timestamp': timestamp,
            # ESP32 polls AI alarm via this POST response
            'ai_high_risk': _is_ai_alarm_active(),
            'ai_alarm_id': ai_alarm_state.get('alarm_id', 0),
            'ai_top_prediction': ai_alarm_state.get('top_prediction'),
            'ai_is_safe': ai_alarm_state.get('is_safe'),
            'ai_final_score': ai_alarm_state.get('final_score'),
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/alarm/ai', methods=['GET'])
def get_ai_alarm():
    """ESP32 / debug: latest AI HIGH-RISK alarm state from image analysis."""
    return jsonify({
        'ai_high_risk': _is_ai_alarm_active(),
        'ai_alarm_id': ai_alarm_state.get('alarm_id', 0),
        'top_prediction': ai_alarm_state.get('top_prediction'),
        'final_score': ai_alarm_state.get('final_score'),
        'is_safe': ai_alarm_state.get('is_safe'),
        'timestamp': ai_alarm_state.get('timestamp'),
    })


@app.route('/api/alarm/test', methods=['POST'])
def test_alarm():
    """Manually trigger AI alarm for 15 seconds to test ESP32 buzzer and LED."""
    global ai_alarm_state
    now_ts = time.time()
    current_alarm_id = ai_alarm_state.get('alarm_id', 0) + 1
    ai_alarm_state = {
        'ai_high_risk': True,
        'top_prediction': 'TEST_MANUAL_ALARM',
        'final_score': 0.99,
        'is_safe': False,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'alarm_id': current_alarm_id,
        'alarm_until': now_ts + 15.0,
    }
    print(f"[AI-ALARM TEST] Triggered manual test alarm_id={current_alarm_id}", flush=True)
    return jsonify({
        'success': True,
        'message': 'Test alarm activated for 15 seconds',
        'ai_alarm_id': current_alarm_id,
        'ai_high_risk': True,
    })


def _empty_sensor_payload(status='no_data'):
    return {
        'device_id': None,
        'gas': None,
        'temperature': None,
        'humidity': None,
        'mq135_raw': None,
        'mq2_digital': None,
        'alarm': None,
        'timestamp': None,
        'age_seconds': None,
        'status': status
    }


def _purge_stale_sensor_entries():
    """Retained for compatibility; persistent readings must never be purged."""
    return None


@app.route('/api/sensors/latest', methods=['GET'])
def get_latest_sensor_data():
    """Get the latest sensor data for the frontend to display."""
    device_id = request.args.get('device_id', None)
    try:
        query = SensorReading.query
        if device_id:
            query = query.filter_by(device_id=device_id)
            reading = query.order_by(SensorReading.timestamp.desc()).first()
        else:
            hardware_query = query.filter_by(source='hardware')
            reading = hardware_query.order_by(SensorReading.timestamp.desc()).first()
            if reading is None:
                reading = query.filter_by(source='simulation').order_by(
                    SensorReading.timestamp.desc()
                ).first()
        if reading is not None:
            payload = _sensor_reading_payload(reading)
            status, age = _get_device_status_with_age(payload['timestamp'])
            return jsonify({**payload, 'status': status, 'age_seconds': age})
    except Exception as db_error:
        db.session.rollback()
        print(f"[DATABASE] Could not read latest sensor data: {db_error}", flush=True)

    readings = list(sensor_data_store.values())
    if device_id:
        readings = [entry for entry in readings if entry['device_id'] == device_id]
    else:
        hardware_readings = [entry for entry in readings if entry.get('source') == 'hardware']
        readings = hardware_readings or [
            entry for entry in readings if entry.get('source') == 'simulation'
        ]
    if readings:
        latest = max(readings, key=lambda x: x['timestamp'])
        status, age = _get_device_status_with_age(latest['timestamp'])
        return jsonify({**latest, 'status': status, 'age_seconds': age})
    return jsonify(_empty_sensor_payload('no_data'))


@app.route('/api/sensors/analyze', methods=['POST'])
def analyze_sensors():
    """Analyze food safety risk using sensor data only (no image required).
    Used by both Demo Mode and Real Mode."""
    global engine
    if engine is None:
        engine = RiskEngine()
    
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON payload'}), 400
        
        # Validate required fields
        try:
            temperature = float(data.get('temperature', 25.0))
            humidity = float(data.get('humidity', 60.0))
            gas = float(data.get('gas', 100.0))
        except (ValueError, TypeError):
            return jsonify({'error': 'Sensor values must be numeric'}), 400
        
        category = data.get('category', None)
        result = engine.analyze_sensors_only(
            temperature,
            humidity,
            gas,
            data.get('mq135_raw'),
            data.get('mq2_digital'),
            data.get('alarm'),
            category=category,
        )
        _update_ai_alarm_from_result(result)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sensors/status', methods=['GET'])
def get_sensor_status():
    """Check ESP32 device connection status."""
    device_id = request.args.get('device_id', None)
    try:
        query = Device.query
        if device_id:
            query = query.filter_by(device_id=device_id)
            device = query.first()
            if device is None:
                return jsonify({'device_id': device_id, 'status': 'no_data', 'last_seen': None})
            status = _get_device_status_with_age(_utc_iso(device.last_seen))[0]
            device.status = status
            db.session.commit()
            return jsonify({'device_id': device_id, 'status': status, 'last_seen': _utc_iso(device.last_seen)})

        devices = []
        for device in query.order_by(Device.device_id).all():
            status = _get_device_status_with_age(_utc_iso(device.last_seen))[0]
            devices.append({'device_id': device.device_id, 'status': status, 'last_seen': _utc_iso(device.last_seen)})
        if devices:
            return jsonify({'devices': devices})
    except Exception as db_error:
        db.session.rollback()
        print(f"[DATABASE] Could not read device status: {db_error}", flush=True)

    if device_id and device_id in sensor_data_store:
        entry = sensor_data_store[device_id]
        return jsonify({'device_id': device_id, 'status': _get_device_status(entry['timestamp']), 'last_seen': entry['timestamp']})
    if not sensor_data_store:
        return jsonify({'device_id': None, 'status': 'no_data', 'last_seen': None})
    return jsonify({'devices': [
        {'device_id': did, 'status': _get_device_status(entry['timestamp']), 'last_seen': entry['timestamp']}
        for did, entry in sensor_data_store.items()
    ]})


@app.route('/api/sensors/history', methods=['GET'])
def get_sensor_history():
    device_id = request.args.get('device_id')
    try:
        limit = min(max(int(request.args.get('limit', 100)), 1), 1000)
    except (TypeError, ValueError):
        limit = 100
    query = SensorReading.query
    if device_id:
        query = query.filter_by(device_id=device_id)
    readings = query.order_by(SensorReading.timestamp.desc()).limit(limit).all()
    return jsonify({
        'success': True,
        'device_id': device_id,
        'readings': [_sensor_reading_payload(reading) for reading in readings],
    })


@app.route('/api/risk/history', methods=['GET'])
def get_risk_history():
    device_id = request.args.get('device_id')
    try:
        limit = min(max(int(request.args.get('limit', 50)), 1), 1000)
    except (TypeError, ValueError):
        limit = 50
    query = RiskAnalysis.query
    if device_id:
        query = query.filter_by(device_id=device_id)
    total_count = query.count()
    analyses = query.order_by(RiskAnalysis.timestamp.desc()).limit(limit).all()
    return jsonify({
        'success': True,
        'device_id': device_id,
        'total_count': total_count,
        'analyses': [{
            'id': analysis.id,
            'device_id': analysis.device_id,
            'image_filename': analysis.image_filename,
            'image_path': analysis.image_path,
            'image_available': bool(analysis.image_path or analysis.image_data),
            'image_prediction': analysis.image_prediction,
            'image_score': analysis.image_score,
            'sensor_score': analysis.sensor_score,
            'final_score': analysis.final_score,
            'is_safe': analysis.is_safe,
            'ai_high_risk': analysis.ai_high_risk,
            'timestamp': _utc_iso(analysis.timestamp),
        } for analysis in analyses],
    })


MOBILENET_TRAINED_ITEMS = [
    {'name': 'Apples', 'key': 'apple', 'icon': '🍎', 'dataset_class': 'freshapples / rottenapples'},
    {'name': 'Banana', 'key': 'banana', 'icon': '🍌', 'dataset_class': 'freshbanana / rottenbanana'},
    {'name': 'Bitter Gourd', 'key': 'bitter', 'icon': '🥒', 'dataset_class': 'freshbittergroud / rottenbittergroud'},
    {'name': 'Capsicum', 'key': 'capsicum', 'icon': '🫑', 'dataset_class': 'freshcapsicum / rottencapsicum'},
    {'name': 'Cucumber', 'key': 'cucumber', 'icon': '🥒', 'dataset_class': 'freshcucumber / rottencucumber'},
    {'name': 'Okra', 'key': 'okra', 'icon': '🥬', 'dataset_class': 'freshokra / rottenokra'},
    {'name': 'Oranges', 'key': 'orange', 'icon': '🍊', 'dataset_class': 'freshoranges / rottenoranges'},
    {'name': 'Potato', 'key': 'potato', 'icon': '🥔', 'dataset_class': 'freshpotato / rottenpotato'},
    {'name': 'Tomato', 'key': 'tomato', 'icon': '🍅', 'dataset_class': 'freshtomato / rottentomato'},
]


@app.route('/api/risk/stats', methods=['GET'])
def get_risk_stats():
    """Return persistent scan aggregates, dataset food items, and recent scan logs from database."""
    device_id = request.args.get('device_id')
    query = RiskAnalysis.query
    if device_id:
        query = query.filter_by(device_id=device_id)
    total_scans = query.count()

    items_map = {}
    for item in MOBILENET_TRAINED_ITEMS:
        items_map[item['key']] = {
            'name': item['name'],
            'key': item['key'],
            'icon': item['icon'],
            'dataset_class': item['dataset_class'],
            'total': 0,
            'fresh': 0,
            'rotten': 0,
            'safety_rate': None,
            'status': 'NO SCANS',
            'last_scanned': None,
        }

    analyses = query.order_by(RiskAnalysis.timestamp.desc()).all()
    safe_count = 0
    at_risk_count = 0
    unsafe_count = 0

    for a in analyses:
        score = a.final_score if a.final_score is not None else (0.1 if a.is_safe else 0.8)
        if not a.is_safe or score >= 0.7:
            unsafe_count += 1
            is_rotten = True
        elif score >= 0.4:
            at_risk_count += 1
            is_rotten = False
        else:
            safe_count += 1
            is_rotten = False

        pred = (a.image_prediction or '').lower().replace('_', ' ')
        matched_key = None
        for item in MOBILENET_TRAINED_ITEMS:
            if item['key'] in pred or (item['key'] + 's') in pred:
                matched_key = item['key']
                break

        if not matched_key and pred:
            clean_name = pred.replace('fresh', '').replace('rotten', '').strip()
            if clean_name:
                token = clean_name.split()[0]
                matched_key = token
                if matched_key not in items_map:
                    items_map[matched_key] = {
                        'name': clean_name.capitalize(),
                        'key': matched_key,
                        'icon': '📦',
                        'dataset_class': f'fresh_{matched_key} / rotten_{matched_key}',
                        'total': 0,
                        'fresh': 0,
                        'rotten': 0,
                        'safety_rate': None,
                        'status': 'NO SCANS',
                        'last_scanned': None,
                    }

        if matched_key and matched_key in items_map:
            entry = items_map[matched_key]
            entry['total'] += 1
            if a.is_safe and not is_rotten:
                entry['fresh'] += 1
            else:
                entry['rotten'] += 1
            if not entry['last_scanned']:
                entry['last_scanned'] = _utc_iso(a.timestamp)

    food_items = []
    for item in items_map.values():
        if item['total'] > 0:
            rate = round((item['fresh'] / item['total']) * 100, 1)
            item['safety_rate'] = rate
            if rate == 100.0:
                item['status'] = 'SAFE'
            elif rate >= 50.0:
                item['status'] = 'WARNING'
            else:
                item['status'] = 'SPOILED'
        else:
            item['safety_rate'] = None
            item['status'] = 'NO SCANS'
        food_items.append(item)

    recent_scans = []
    for a in analyses[:20]:
        pred = a.image_prediction or 'Food analysis'
        is_safe = bool(a.is_safe)
        score_val = a.final_score if a.final_score is not None else 0.0
        score_pct = round(score_val * 100, 1)
        if not is_safe or score_val >= 0.7:
            status = 'UNSAFE'
        elif score_val >= 0.4:
            status = 'WARNING'
        else:
            status = 'SAFE'

        recent_scans.append({
            'id': a.id,
            'prediction': pred.replace('_', ' '),
            'raw_prediction': pred,
            'is_safe': is_safe,
            'final_score': score_val,
            'score_pct': score_pct,
            'status': status,
            'timestamp': _utc_iso(a.timestamp),
        })

    return jsonify({
        'success': True,
        'total_scans': total_scans,
        'safe_count': safe_count,
        'at_risk_count': at_risk_count,
        'unsafe_count': unsafe_count,
        'safe_percent': round((safe_count / total_scans) * 100, 1) if total_scans else 0.0,
        'risk_percent': round((at_risk_count / total_scans) * 100, 1) if total_scans else 0.0,
        'unsafe_percent': round((unsafe_count / total_scans) * 100, 1) if total_scans else 0.0,
        'food_items': food_items,
        'recent_scans': recent_scans,
    })


@app.route('/api/risk/images/<int:analysis_id>', methods=['GET'])
def get_risk_image(analysis_id):
    analysis = db.session.get(RiskAnalysis, analysis_id)
    if analysis is None:
        return jsonify({'success': False, 'error': 'Image not found'}), 404
    if analysis.image_path:
        stored_path = (BASE_DIR / analysis.image_path).resolve()
        if stored_path.parent == RESULT_IMAGES_DIR.resolve() and stored_path.is_file():
            return send_file(
                stored_path,
                mimetype=analysis.image_mime_type or 'application/octet-stream',
                download_name=analysis.image_filename or stored_path.name,
            )
    if analysis.image_data is None:
        return jsonify({'success': False, 'error': 'Image not found'}), 404
    return send_file(
        io.BytesIO(analysis.image_data),
        mimetype=analysis.image_mime_type or 'application/octet-stream',
        download_name=analysis.image_filename or f'analysis-{analysis_id}.bin',
    )


@app.route('/api/sensors/danger_zone', methods=['GET'])
def get_danger_zone():
    """Return cumulative danger-zone minutes for a device (or the most recent device).
    Optional query param: ?device_id=<id>"""
    device_id = request.args.get('device_id', None)
    try:
        query = DangerZoneLog.query
        if device_id:
            query = query.filter_by(device_id=device_id)
        latest = query.order_by(DangerZoneLog.timestamp.desc()).first()
        if latest is None:
            return jsonify({
                'device_id': device_id,
                'cumulative_minutes': 0.0,
                'hours_in_danger_zone': 0.0,
                'last_violation': None,
                'status': 'no_data',
            })
        cumulative_min = latest.cumulative_minutes
        # Find timestamp of last reading that was actually IN the zone
        last_violation_row = (
            DangerZoneLog.query
            .filter(
                DangerZoneLog.in_danger_zone == True,
                *([] if not device_id else [DangerZoneLog.device_id == device_id]),
            )
            .order_by(DangerZoneLog.timestamp.desc())
            .first()
        )
        alert_level = 'safe'
        if cumulative_min >= 120:
            alert_level = 'critical'
        elif cumulative_min >= 30:
            alert_level = 'warning'

        return jsonify({
            'device_id': latest.device_id,
            'cumulative_minutes': round(cumulative_min, 2),
            'hours_in_danger_zone': round(cumulative_min / 60.0, 3),
            'last_violation': _utc_iso(last_violation_row.timestamp) if last_violation_row else None,
            'alert_level': alert_level,
            'status': 'ok',
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/metrics', methods=['GET'])
def get_model_metrics():
    """Return verified benchmark metrics from 06_results/metrics.json (Phase 2.3)."""
    metrics_file = BASE_DIR / '06_results' / 'metrics.json'
    if not metrics_file.exists():
        return jsonify({
            'success': False,
            'error': 'Metrics not generated yet. Run evaluate_model.py first.',
        }), 404
    try:
        data = json.loads(metrics_file.read_text(encoding='utf-8'))
        return jsonify({
            'success': True,
            'metrics': data,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _get_device_status_with_age(timestamp_str):
    """Return (status, age_seconds) from last server-side reading timestamp."""
    try:
        last_time = datetime.fromisoformat(timestamp_str)
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_seconds = (now - last_time).total_seconds()

        if age_seconds < SENSOR_TIMEOUT_SECONDS:
            return 'connected', round(age_seconds, 1)
        return 'disconnected', round(age_seconds, 1)
    except Exception:
        return 'unknown', None


def _get_device_status(timestamp_str):
    """Determine device connection status based on last data timestamp."""
    status, _age = _get_device_status_with_age(timestamp_str)
    return status


if __name__ == '__main__':
    try:
        engine = RiskEngine()
    except Exception as e:
        print("Warning: Engine initialization failed, will retry on first request.", e)

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)
