"""
Configuration management system
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        load_dotenv()  # Load environment variables
        
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "main_config.yml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._substitute_env_vars()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _substitute_env_vars(self) -> None:
        """Substitute environment variables in configuration"""
        def _substitute(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    obj[key] = _substitute(value)
            elif isinstance(obj, list):
                obj = [_substitute(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                obj = os.environ.get(env_var, obj)
            return obj
        
        self.config = _substitute(self.config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key
        
        Args:
            key: Dot-separated key path (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value
        
        Args:
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file
        
        Args:
            path: Path to save configuration
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    @property
    def mode(self) -> str:
        """Get current operational mode"""
        return self.get('system.mode', 'simulation')
    
    @property
    def is_simulation(self) -> bool:
        """Check if in simulation mode"""
        return self.mode == 'simulation'
    
    @property
    def is_paper(self) -> bool:
        """Check if in paper trading mode"""
        return self.mode == 'paper'
    
    @property
    def is_live(self) -> bool:
        """Check if in live trading mode"""
        return self.mode == 'live'