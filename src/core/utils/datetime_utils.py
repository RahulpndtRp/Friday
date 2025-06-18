from datetime import datetime, timezone
from typing import Optional, Union


def utc_now() -> datetime:
    """Get current UTC datetime - Python 3.12 compatible."""
    return datetime.now(timezone.utc)


def safe_parse_datetime(dt_value: Union[str, datetime, None]) -> Optional[datetime]:
    """Safely parse datetime from various formats - Python 3.12 compatible."""
    if not dt_value:
        return None

    if isinstance(dt_value, datetime):
        return dt_value

    if isinstance(dt_value, str):
        try:
            # Handle different ISO format variations
            if dt_value.endswith("Z"):
                dt_value = dt_value[:-1] + "+00:00"

            # Try parsing with timezone info
            if "+" in dt_value or dt_value.endswith("Z"):
                return datetime.fromisoformat(dt_value)
            else:
                # Assume UTC if no timezone info
                dt = datetime.fromisoformat(dt_value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt

        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse datetime '{dt_value}': {e}")
            return utc_now()

    return None


def to_iso(dt: datetime) -> str:
    """Convert datetime to ISO format string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()
