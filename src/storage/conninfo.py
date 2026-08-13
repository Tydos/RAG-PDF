"""Connection string helpers."""

import socket
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def prefer_ipv4_conninfo(url: str) -> str:
    """Resolve the host to IPv4 and set hostaddr (fixes WSL2 IPv6 unreachable errors)."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host or host in ("localhost", "127.0.0.1"):
        return url

    try:
        addrs = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return url

    if not addrs:
        return url

    ipv4 = addrs[0][4][0]
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if query.get("hostaddr") == ipv4:
        return url
    query["hostaddr"] = ipv4
    return urlunparse(parsed._replace(query=urlencode(query)))
