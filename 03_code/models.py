from datetime import datetime, timezone

from database import db


def utc_now():
    return datetime.now(timezone.utc)


class SensorReading(db.Model):
    __tablename__ = 'sensor_readings'

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(100), nullable=False, index=True)
    source = db.Column(db.String(20), nullable=False, default='hardware')
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    gas = db.Column(db.Float, nullable=False)
    mq135_raw = db.Column(db.Integer, nullable=True)
    mq2_digital = db.Column(db.Integer, nullable=True)
    alarm = db.Column(db.Boolean, nullable=True)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, index=True, default=utc_now)


class RiskAnalysis(db.Model):
    __tablename__ = 'risk_analysis'

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(100), nullable=True, index=True)
    image_filename = db.Column(db.String(255), nullable=True)
    image_mime_type = db.Column(db.String(100), nullable=True)
    image_data = db.Column(db.LargeBinary, nullable=True)
    image_path = db.Column(db.String(500), nullable=True)
    image_prediction = db.Column(db.String(255), nullable=True)
    image_score = db.Column(db.Float, nullable=True)
    sensor_score = db.Column(db.Float, nullable=True)
    final_score = db.Column(db.Float, nullable=True)
    is_safe = db.Column(db.Boolean, nullable=True)
    ai_high_risk = db.Column(db.Boolean, nullable=True)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, index=True, default=utc_now)


class Device(db.Model):
    __tablename__ = 'devices'

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(100), nullable=False, unique=True, index=True)
    name = db.Column(db.String(255), nullable=True)
    last_seen = db.Column(db.DateTime(timezone=True), nullable=True, index=True)
    status = db.Column(db.String(32), nullable=False, default='no_data')
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)


class DangerZoneLog(db.Model):
    """Tracks each sensor reading that falls inside the bacterial danger zone (5°C – 60°C).
    cumulative_minutes holds the running total for that device up to this reading."""
    __tablename__ = 'danger_zone_log'

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(100), nullable=False, index=True)
    temperature = db.Column(db.Float, nullable=False)
    in_danger_zone = db.Column(db.Boolean, nullable=False, default=False)
    # Running cumulative minutes the device has spent in the danger zone
    cumulative_minutes = db.Column(db.Float, nullable=False, default=0.0)
    timestamp = db.Column(
        db.DateTime(timezone=True), nullable=False, index=True, default=utc_now
    )