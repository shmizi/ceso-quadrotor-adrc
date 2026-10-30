import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODES = ("pid", "eso", "ceso", "tanh", "td3")
TRAJECTORIES = ("hover", "circle", "figure8", "waypoints")


def load_config(path=None):
    with open(path or ROOT / "config.json", encoding="utf-8") as stream:
        return json.load(stream)


def td3_settings(c):
    return c["td3_"+c["td3_variant"]]
