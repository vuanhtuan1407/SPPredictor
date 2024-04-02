import os.path
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def abspath(path: str):
    return str(Path(ROOT_DIR) / path)


def abspaths(paths: list[str]):
    return [abspath(path) for path in paths]
