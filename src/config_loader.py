import os
import yaml

# Automatically load environment variables from .env file at project root if available
try:
    from dotenv import load_dotenv
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)
    _env_path = os.path.join(_project_root, ".env")
    if os.path.exists(_env_path):
        load_dotenv(dotenv_path=_env_path, override=True)
    else:
        load_dotenv()
except ImportError:
    pass

def load_config(config_path=None):
    if config_path is None:
        # Determine the project root relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, "config", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths in the config to absolute paths based on project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Resolve vector_store persist_directory
    if "vector_store" in config and "persist_directory" in config["vector_store"]:
        dir_path = config["vector_store"]["persist_directory"]
        if not os.path.isabs(dir_path):
            config["vector_store"]["persist_directory"] = os.path.join(project_root, dir_path)

    # 2. Resolve database connection_string if it's sqlite
    if "database" in config and "connection_string" in config["database"]:
        conn_str = config["database"]["connection_string"]
        if conn_str.startswith("sqlite:///"):
            db_rel_path = conn_str.replace("sqlite:///", "")
            if not os.path.isabs(db_rel_path):
                db_abs_path = os.path.join(project_root, db_rel_path)
                config["database"]["connection_string"] = f"sqlite:///{db_abs_path}"

    return config

def get_active_provider(config: dict) -> str:
    return config.get("llm", {}).get("provider", "gemini").lower()

def get_api_key_env_var_name(config: dict) -> str:
    provider = get_active_provider(config)
    if provider == "openai":
        return "OPENAI_API_KEY"
    elif provider == "groq":
        return "GROQ_API_KEY"
    elif provider == "nvidia":
        return "NVIDIA_API_KEY"
    elif provider == "openrouter":
        return "OPENROUTER_API_KEY"
    else:
        return "GEMINI_API_KEY"

def clean_api_key(key: str) -> str:
    if not key:
        return ""
    key = str(key).strip()
    quotes = ['"', "'", '“', '”', '‘', '’']
    while len(key) > 0 and (key[0] in quotes or key[-1] in quotes):
        if key[0] in quotes:
            key = key[1:]
        if len(key) > 0 and key[-1] in quotes:
            key = key[:-1]
    return key.strip()

def is_api_key_set(config: dict) -> bool:
    name = get_api_key_env_var_name(config)
    return bool(clean_api_key(os.environ.get(name, "")))

def get_openrouter_api_key(config: dict = None) -> str:
    """Retrieve OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    return clean_api_key(key)

def is_openrouter_api_key_set(config: dict = None) -> bool:
    """Check if OpenRouter API key is set."""
    return bool(get_openrouter_api_key(config))

def is_jev_enabled(config: dict) -> bool:
    """Check if either intent or fraud classifier is configured to use JEV."""
    classifiers = config.get("classifiers", {})
    return classifiers.get("intent") == "jev" or classifiers.get("fraud") == "jev"
