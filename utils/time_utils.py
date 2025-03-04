from datetime import datetime, timezone, UTC

def get_utc_now():
    """Get current UTC time in a timezone-aware manner"""
    return datetime.now(UTC)

def timestamp_to_datetime(timestamp):
    """Convert timestamp to timezone-aware datetime"""
    return datetime.fromtimestamp(timestamp, tz=UTC)

def datetime_to_timestamp(dt):
    """Convert datetime to UTC timestamp"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp() 