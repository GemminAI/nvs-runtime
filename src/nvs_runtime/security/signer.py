"""Deterministic HMAC signer for capability tokens."""

from __future__ import annotations

import hashlib
import hmac


class HmacSigner:
    """Test / default signer — no external crypto dependency required."""

    def __init__(self, secret: bytes = b"nvs-runtime-dev-secret") -> None:
        self._secret = secret

    def sign(self, payload: bytes) -> bytes:
        return hmac.new(self._secret, payload, hashlib.sha256).digest()

    def verify(self, payload: bytes, signature: bytes) -> bool:
        expected = self.sign(payload)
        return hmac.compare_digest(expected, signature)
