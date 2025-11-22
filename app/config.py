import getpass
import json
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
ADDRESS_ALIAS_PATH = PROJECT_ROOT / "data" / "address_aliases.json"


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


def load_address_aliases() -> dict[str, str]:
    """
    Load user-defined address aliases (natural language -> dataset property names).
    Returns an empty mapping when no alias file is provided.
    """

    env_override = os.getenv("ADDRESS_ALIAS_FILE")
    path = Path(env_override) if env_override else ADDRESS_ALIAS_PATH
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse address alias file {path}") from exc

    normalized = {}
    for raw_key, value in data.items():
        if not raw_key or not value:
            continue
        normalized[str(raw_key).lower()] = str(value)
    return normalized


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Env path: {ENV_PATH}")

    load_env()

    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("GEMINI_API_KEY")
    # _set_if_undefined("ANTHROPIC_API_KEY")
    # _set_if_undefined("TAVILY_API_KEY")
