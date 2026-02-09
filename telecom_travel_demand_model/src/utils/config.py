"""Configuration management for the telecom travel demand model."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class Config:
    """
    Configuration manager for the telecom travel demand model.

    Loads configuration from YAML files and provides easy access to parameters.
    Supports environment variable overrides and nested parameter access.

    Example:
        >>> config = Config("config/config.yaml")
        >>> distance_threshold = config.get("stay_detection.distance_threshold", default=500)
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to configuration YAML file. If None, uses default config.
        """
        self._config: Dict[str, Any] = {}
        self._config_path = config_path

        if config_path:
            self.load(config_path)
        else:
            # Load default configuration
            default_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
            if default_path.exists():
                self.load(default_path)

    def load(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)
            # Handle empty YAML files
            self._config = loaded if loaded is not None else {}

        self._config_path = config_path
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_prefix = "TELECOM_TDM_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Convert TELECOM_TDM_STAY_DETECTION_DISTANCE_THRESHOLD to
                # stay_detection.distance_threshold
                config_key = key[len(env_prefix):].lower().replace("_", ".")
                self.set(config_key, self._parse_env_value(value))

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON for complex types
        import json
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Return as string
            return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "stay_detection.distance_threshold")
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key.

        Args:
            key: Dot-notation key.
            value: Value to set.
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            # Handle case where intermediate key exists but is not a dict
            if k not in config or not isinstance(config.get(k), dict):
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def __getitem__(self, key: str) -> Any:
        """Get configuration value."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output path. If None, overwrites original config file.
        """
        path = Path(path) if path else self._config_path
        if path is None:
            raise ValueError("No path specified for saving configuration")

        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    # Convenience properties for commonly accessed parameters
    @property
    def stay_detection(self) -> Dict[str, Any]:
        """Get stay detection configuration."""
        return self.get("stay_detection", {})

    @property
    def home_work_inference(self) -> Dict[str, Any]:
        """Get home/work inference configuration."""
        return self.get("home_work_inference", {})

    @property
    def trip_generation(self) -> Dict[str, Any]:
        """Get trip generation configuration."""
        return self.get("trip_generation", {})

    @property
    def od_matrix(self) -> Dict[str, Any]:
        """Get OD matrix configuration."""
        return self.get("od_matrix", {})

    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.get("preprocessing", {})

    @property
    def data_fusion(self) -> Dict[str, Any]:
        """Get data fusion configuration."""
        return self.get("data_fusion", {})


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _global_config
    _global_config = config
