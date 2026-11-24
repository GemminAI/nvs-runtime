"""Adapters that map nvs-runtime's protocol-only boundary onto a specific sibling
service's real wire format (e.g. ``hekb_vnext.py`` -> hekb-vnext's ``/v3/*`` HTTP API).
Not providers themselves -- providers (``nvs_runtime.providers``) depend on these only
through the protocols in ``nvs_runtime.core.protocols`` (e.g. ``ObjectStore``)."""
