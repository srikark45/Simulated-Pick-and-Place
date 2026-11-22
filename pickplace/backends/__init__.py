from __future__ import annotations


def make_backend(name: str):
    if name == "mock":
        from .mock import MockBackend
        return MockBackend()
    if name == "mujoco":
        from .mujoco_backend import MuJoCoBackend
        return MuJoCoBackend()
    raise ValueError(f"unknown backend {name!r}")
