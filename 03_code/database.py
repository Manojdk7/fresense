from pathlib import Path

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text


BASE_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = BASE_DIR / 'foodsafe.db'

db = SQLAlchemy()


def init_database(app):
    app.config.setdefault('SQLALCHEMY_DATABASE_URI', f'sqlite:///{DATABASE_PATH}')
    app.config.setdefault('SQLALCHEMY_TRACK_MODIFICATIONS', False)
    db.init_app(app)
    with app.app_context():
        db.create_all()
        _migrate_sensor_reading_source()
        _migrate_risk_analysis_image_columns()
        _migrate_danger_zone_log()


def _migrate_sensor_reading_source():
    columns = {column['name'] for column in inspect(db.engine).get_columns('sensor_readings')}
    if 'source' not in columns:
        db.session.execute(text(
            "ALTER TABLE sensor_readings ADD COLUMN source VARCHAR(20) "
            "NOT NULL DEFAULT 'legacy'"
        ))
        db.session.commit()


def _migrate_risk_analysis_image_columns():
    columns = {column['name'] for column in inspect(db.engine).get_columns('risk_analysis')}
    new_columns = {
        'image_filename': 'VARCHAR(255)',
        'image_mime_type': 'VARCHAR(100)',
        'image_data': 'BLOB',
        'image_path': 'VARCHAR(500)',
    }
    for column_name, column_type in new_columns.items():
        if column_name not in columns:
            db.session.execute(text(
                f'ALTER TABLE risk_analysis ADD COLUMN {column_name} {column_type}'
            ))
    db.session.commit()


def _migrate_danger_zone_log():
    """Ensure danger_zone_log table exists on databases created before Phase 1."""
    inspector = inspect(db.engine)
    existing_tables = inspector.get_table_names()
    if 'danger_zone_log' not in existing_tables:
        db.session.execute(text(
            'CREATE TABLE IF NOT EXISTS danger_zone_log ('
            '  id INTEGER PRIMARY KEY AUTOINCREMENT,'
            '  device_id VARCHAR(100) NOT NULL,'
            '  temperature FLOAT NOT NULL,'
            '  in_danger_zone BOOLEAN NOT NULL DEFAULT 0,'
            '  cumulative_minutes FLOAT NOT NULL DEFAULT 0.0,'
            '  timestamp DATETIME NOT NULL'
            ')'
        ))
        db.session.execute(text(
            'CREATE INDEX IF NOT EXISTS ix_danger_zone_log_device_id'
            ' ON danger_zone_log (device_id)'
        ))
        db.session.execute(text(
            'CREATE INDEX IF NOT EXISTS ix_danger_zone_log_timestamp'
            ' ON danger_zone_log (timestamp)'
        ))
        db.session.commit()
        print('[DB] Created danger_zone_log table.')
