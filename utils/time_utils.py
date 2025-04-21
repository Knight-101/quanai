from datetime import datetime, timezone

def get_utc_now():
    """Get current UTC time in a timezone-aware manner"""
    return datetime.now(timezone.utc)

def timestamp_to_datetime(timestamp):
    """Convert timestamp to timezone-aware datetime"""
    dt = datetime.fromtimestamp(timestamp)
    return dt.replace(tzinfo=timezone.utc)

def datetime_to_timestamp(dt):
    """Convert datetime to UTC timestamp"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() 