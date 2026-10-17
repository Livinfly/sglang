"""Local control socket protocol, used only outside client E2E."""

import json
import socket
from contextlib import suppress


def send(stream, value):
    stream.write((json.dumps(value, default=str) + "\n").encode())
    stream.flush()


def receive(stream):
    line = stream.readline()
    if not line:
        raise RuntimeError("Benchmark observer disconnected")
    value = json.loads(line)
    if "error" in value:
        raise RuntimeError(value["error"])
    return value


def connect(directory, world):
    clients = []
    try:
        for rank in range(world):
            sock = socket.socket(socket.AF_UNIX)
            sock.settimeout(7200)
            sock.connect(str(directory / f"{rank}.sock"))
            clients.append((sock, sock.makefile("rwb")))
        return clients
    except BaseException:
        close_clients(clients)
        raise


def close_clients(clients):
    """A broken observer must not prevent closing peers or shutting down workers."""
    for sock, stream in clients:
        with suppress(OSError, ValueError):
            send(stream, dict(command="close"))
        with suppress(OSError, ValueError):
            stream.close()
        with suppress(OSError):
            sock.close()


def exchange(clients, command):
    # Send to ALL ranks before receiving: ARM/collect include an all-rank barrier.
    for _, stream in clients:
        send(stream, command)
    return [receive(stream) for _, stream in clients]
