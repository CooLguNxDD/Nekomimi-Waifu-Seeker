"""Best Waifu Determination Engine — feature search + Laya decisions."""

from . import envfile

__version__ = "0.2.0"

# Before any module reads its settings: fill unset variables from ``.env``.
envfile.load()
