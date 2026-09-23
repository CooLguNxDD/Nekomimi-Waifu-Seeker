"""Suite-wide setup: never read a developer's ``.env`` (it may hold real keys
and switch on billed services); every test sets the variables it needs."""

import os

os.environ["WAIFU_ENV_FILE"] = "0"
