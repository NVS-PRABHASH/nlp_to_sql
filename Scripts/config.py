import os
import sys
import yaml
import logging
from logging.config import dictConfig
from pathlib import Path
from functools import lru_cache

CONFIG_PATH = Path(__file__).parent.parent / "Config/config.yaml"
DEFAULT_LOG_FILE = Path(__file__).parent.parent / "logs/ai_insight.log"


@lru_cache(maxsize=1)
def get_config() -> dict:
    """Load and cache application configuration from config.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _ensure_log_dirs(cfg: dict) -> None:
    """Ensure directories in logging handlers exist before configuring logging."""
    try:
        if "handlers" not in cfg.get("logging", {}):
            return
            
        for handler_name, handler_cfg in cfg["logging"]["handlers"].items():
            if "filename" not in handler_cfg:
                continue

            original_filename = handler_cfg.get("filename")
            try:
                # Resolve to absolute path within project if relative
                log_path = Path(original_filename).expanduser()
                if not log_path.is_absolute():
                    log_path = (Path(__file__).parent.parent / log_path).resolve()

                try:
                    # Try to create directory and file
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        log_path.parent.chmod(0o755)
                    except Exception:
                        # Permission change is best-effort
                        pass
                    if not log_path.exists():
                        log_path.touch(mode=0o644)
                    handler_cfg["filename"] = str(log_path)
                except Exception as e_dir:
                    # Fallback to default location under project logs/
                    fallback = DEFAULT_LOG_FILE
                    try:
                        fallback.parent.mkdir(parents=True, exist_ok=True)
                        if not fallback.exists():
                            fallback.touch(mode=0o644)
                        handler_cfg["filename"] = str(fallback)
                        print(
                            f"Warning: Using fallback log path {fallback} for handler '{handler_name}': {e_dir}",
                            file=sys.stderr,
                        )
                    except Exception as e_fb:
                        print(
                            f"Warning: Failed to set up fallback log file {fallback}: {e_fb}",
                            file=sys.stderr,
                        )
            except Exception as e_path:
                # If even resolving/expanding fails, force fallback
                fallback = DEFAULT_LOG_FILE
                try:
                    fallback.parent.mkdir(parents=True, exist_ok=True)
                    if not fallback.exists():
                        fallback.touch(mode=0o644)
                    handler_cfg["filename"] = str(fallback)
                    print(
                        f"Warning: Using fallback log path {fallback} for handler '{handler_name}': {e_path}",
                        file=sys.stderr,
                    )
                except Exception as e_fb2:
                    print(
                        f"Warning: Failed to set up fallback log file {fallback}: {e_fb2}",
                        file=sys.stderr,
                    )
                
    except Exception as e:
        print(f"Error in _ensure_log_dirs: {e}", file=sys.stderr)
        # Don't block startup, but log the error
        import traceback
        traceback.print_exc()


def setup_logging() -> None:
    """
    Configure logging using config.yaml.
    If config loading fails, falls back to basic console logging.
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    try:
        cfg = get_config()
        if "logging" in cfg:
            _ensure_log_dirs(cfg)
            logging.config.dictConfig(cfg["logging"])  # type: ignore
            
            # Ensure the root logger is configured
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            
            # Add handlers if not already added
            if not root_logger.handlers:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                ))
                root_logger.addHandler(console_handler)
                
                # Also add file handler if specified in config
                if 'file' in cfg["logging"].get('handlers', {}):
                    file_cfg = cfg["logging"]["handlers"]["file"]
                    file_handler = logging.FileHandler(
                        filename=file_cfg["filename"],
                        mode=file_cfg.get("mode", "a"),
                        encoding=file_cfg.get("encoding", "utf-8")
                    )
                    file_handler.setFormatter(logging.Formatter(
                        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                    ))
                    file_handler.setLevel(logging.getLevelName(file_cfg.get("level", "INFO")))
                    root_logger.addHandler(file_handler)
            
            return
            
    except Exception as e:
        # Fallback to basic logging if config loading fails
        try:
            # Ensure default log directory exists
            DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        except Exception as _e:
            print(f"Warning: could not create log directory {DEFAULT_LOG_FILE.parent}: {_e}", file=sys.stderr)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    filename=DEFAULT_LOG_FILE,
                    mode='a',
                    encoding='utf-8'
                )
            ]
        )
        logging.error(f"Failed to load logging config: {e}", exc_info=True)
        return


def build_oracle_url(db_cfg: dict) -> str:
    """Build an Oracle SQLAlchemy URL from config dict, respecting the dialect setting."""
    dialect = db_cfg.get("dialect", "oracle+oracledb")  # Default to oracledb (python-oracledb)
    host = db_cfg.get("host")
    port = db_cfg.get("port")
    service = db_cfg.get("service")
    user = db_cfg.get("user")
    password = db_cfg.get("password")
    dsn = f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={host})(PORT={port}))(CONNECT_DATA=(SERVICE_NAME={service})))"
    return f"{dialect}://{user}:{password}@{dsn}"
