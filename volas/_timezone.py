"""First-use Python data resolution; rule evaluation and key caching are Rust."""

from importlib import resources
import os
import zoneinfo


# Keep the interpreter's standard keyed objects alive for the process lifetime,
# even after ZoneInfo's small strong cache evicts them. Configure sources before
# first use and restart the process to update rules.
_zones: dict[str, zoneinfo.ZoneInfo] = {}


def _version(directory: str) -> str | None:
    try:
        with open(os.path.join(directory, "tzdata.zi"), encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("# version "):
                    fields = line.split()
                    return fields[2] if len(fields) > 2 else None
    except (OSError, UnicodeError):
        pass
    return None


def _resolve(key: str) -> tuple[bytes, str, str | None]:
    """Read exactly the first existing file, then the tzdata resource."""
    # Like zoneinfo, normalization may change slash spelling on Windows but
    # must neither change length nor escape the relative search root.
    normalized = os.path.normpath(key)
    base = os.path.normpath(os.path.join("_", "_"))[:-1]
    if (os.path.isabs(key) or len(normalized) != len(key)
            or not os.path.normpath(os.path.join(base, normalized)).startswith(base)):
        raise ValueError(f"unknown timezone {key!r}")
    for directory in zoneinfo.TZPATH:
        path = os.path.join(directory, key)
        if os.path.isfile(path):
            with open(path, "rb") as stream:
                return stream.read(), directory, _version(directory)
    try:
        package = "tzdata.zoneinfo"
        components = key.split("/")
        if len(components) > 1:
            package += "." + ".".join(components[:-1])
        data = resources.files(package).joinpath(components[-1]).read_bytes()
        import tzdata
        return data, "tzdata", getattr(tzdata, "IANA_VERSION", None)
    except (ImportError, FileNotFoundError, UnicodeEncodeError):
        raise ValueError(f"unknown timezone {key!r}") from None


def _pin(key: str) -> zoneinfo.ZoneInfo:
    zone = zoneinfo.ZoneInfo(key)
    _zones[key] = zone
    return zone
