"""Canonical semantic identities shared by fixture and oracle replay."""

import hashlib
import json


def fingerprint(value: object) -> str:
    """Fingerprint canonical JSON semantic content, independent of output location."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
