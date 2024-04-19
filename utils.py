import os.path
from pathlib import Path

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def abspath(path: str):
    return str(Path(_ROOT_DIR, path))


def abspaths(paths: list[str]):
    return [abspath(path) for path in paths]
