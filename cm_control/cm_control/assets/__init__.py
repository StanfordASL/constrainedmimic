import importlib.resources as resources
from pathlib import Path

ASSETS_DIR = Path(resources.files("cm_control.assets"))

G1_ASSETS_DIR = ASSETS_DIR / "unitree_g1"
THIRD_PARTY_DIR = ASSETS_DIR.parent.parent.parent / "third_party"
TWIST2_DIR = THIRD_PARTY_DIR / "TWIST2"
G1_XML = G1_ASSETS_DIR / "cm_g1.xml"

if not TWIST2_DIR.exists():
    raise FileNotFoundError(
        f"TWIST2 repo not located. Expected location: {TWIST2_DIR}\n"
        + "Try re-fetching the git submodules"
    )

CHECKPOINTS_DIR = TWIST2_DIR / "assets" / "ckpts"
