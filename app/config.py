import getpass
import os
from pathlib import Path

from dotenv import load_dotenv


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Unable to locate project root (pyproject.toml not found).")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())

ENV_PATH = PROJECT_ROOT / ".env"


def load_env() -> None:
    if ENV_PATH.exists():
        loaded = load_dotenv(ENV_PATH)
        if not loaded:
            raise RuntimeError(f"Failed to load env file from {ENV_PATH}")
    else:
        raise FileNotFoundError(f"Env file not found at {ENV_PATH}")


def _set_if_undefined(var: str) -> None:
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected file missing: {path}")
    return path


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Env path: {ENV_PATH}")

    load_env()

    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("GEMINI_API_KEY")
    # _set_if_undefined("ANTHROPIC_API_KEY")
    # _set_if_undefined("TAVILY_API_KEY")
