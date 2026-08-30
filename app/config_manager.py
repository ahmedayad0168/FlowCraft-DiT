"""
Configuration management for FlowCraft-DiT production deployment.

This module handles loading and validating configuration from YAML files
and environment variables.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class ModelConfig:
    """Model configuration."""
    checkpoint_path: str = "checkpoints/flowcraft_step10000.pt"
    device: str = "cuda"
    use_ema: bool = True
    dtype: str = "bfloat16"


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    enable_cors: bool = True
    max_request_size: int = 10


@dataclass
class UIConfig:
    """UI configuration."""
    port: int = 8501
    headless: bool = True
    gather_usage_stats: bool = False
    max_upload_size: int = 200
    max_message_size: int = 200


@dataclass
class GenerationConfig:
    """Generation configuration."""
    default_steps: int = 28
    default_cfg_scale: float = 5.0
    default_resolution: int = 128
    max_steps: int = 100
    max_cfg_scale: float = 20.0
    max_resolution: int = 1024
    min_resolution: int = 64


@dataclass
class StorageConfig:
    """Storage configuration."""
    output_dir: str = "outputs"
    api_output_dir: str = "outputs/api"
    ui_output_dir: str = "outputs/ui"
    max_generations: int = 1000
    cleanup_interval: int = 3600


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    enable_torch_compile: bool = False
    enable_cuda_graphs: bool = False
    memory_efficient_attention: bool = False
    allow_tf32: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/flowcraft.log"
    max_bytes: int = 10485760
    backup_count: int = 5


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_auth: bool = False
    api_key_required: bool = False
    rate_limit_enabled: bool = False
    rate_limit_rpm: int = 60
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_metrics: bool = False
    enable_tracing: bool = False
    metrics_port: int = 9090


@dataclass
class ProductionConfig:
    """Main production configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


class ConfigManager:
    """Configuration manager for loading and validating config."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/production.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> ProductionConfig:
        """Load configuration from file and environment variables."""
        # Start with defaults
        config = ProductionConfig()
        
        # Load from YAML if available
        if YAML_AVAILABLE and Path(self.config_path).exists():
            config = self._load_from_yaml(config)
        
        # Override with environment variables
        config = self._load_from_env(config)
        
        # Validate configuration
        self._validate_config(config)
        
        return config
    
    def _load_from_yaml(self, config: ProductionConfig) -> ProductionConfig:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            if not yaml_config:
                return config
            
            # Update model config
            if 'model' in yaml_config:
                config.model = ModelConfig(**yaml_config['model'])
            
            # Update API config
            if 'api' in yaml_config:
                config.api = APIConfig(**yaml_config['api'])
            
            # Update UI config
            if 'ui' in yaml_config:
                config.ui = UIConfig(**yaml_config['ui'])
            
            # Update generation config
            if 'generation' in yaml_config:
                config.generation = GenerationConfig(**yaml_config['generation'])
            
            # Update storage config
            if 'storage' in yaml_config:
                config.storage = StorageConfig(**yaml_config['storage'])
            
            # Update performance config
            if 'performance' in yaml_config:
                config.performance = PerformanceConfig(**yaml_config['performance'])
            
            # Update logging config
            if 'logging' in yaml_config:
                config.logging = LoggingConfig(**yaml_config['logging'])
            
            # Update security config
            if 'security' in yaml_config:
                security_data = yaml_config['security']
                config.security = SecurityConfig(
                    enable_auth=security_data.get('enable_auth', False),
                    api_key_required=security_data.get('api_key_required', False),
                    rate_limit_enabled=security_data.get('rate_limit', {}).get('enabled', False),
                    rate_limit_rpm=security_data.get('rate_limit', {}).get('requests_per_minute', 60),
                    cors_origins=security_data.get('cors_origins', ['*'])
                )
            
            # Update monitoring config
            if 'monitoring' in yaml_config:
                config.monitoring = MonitoringConfig(**yaml_config['monitoring'])
            
            logging.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except Exception as e:
            logging.warning(f"Failed to load YAML config: {e}. Using defaults.")
            return config
    
    def _load_from_env(self, config: ProductionConfig) -> ProductionConfig:
        """Load configuration from environment variables."""
        # Model configuration
        config.model.checkpoint_path = os.getenv('FLOWCRAFT_CHECKPOINT', config.model.checkpoint_path)
        config.model.device = os.getenv('FLOWCRAFT_DEVICE', config.model.device)
        config.model.dtype = os.getenv('FLOWCRAFT_DTYPE', config.model.dtype)
        
        # API configuration
        config.api.host = os.getenv('FLOWCRAFT_API_HOST', config.api.host)
        config.api.port = int(os.getenv('FLOWCRAFT_API_PORT', str(config.api.port)))
        config.api.workers = int(os.getenv('FLOWCRAFT_API_WORKERS', str(config.api.workers)))
        
        # UI configuration
        config.ui.port = int(os.getenv('FLOWCRAFT_UI_PORT', str(config.ui.port)))
        
        # Generation configuration
        config.generation.default_steps = int(os.getenv('FLOWCRAFT_DEFAULT_STEPS', str(config.generation.default_steps)))
        config.generation.default_cfg_scale = float(os.getenv('FLOWCRAFT_DEFAULT_CFG', str(config.generation.default_cfg_scale)))
        config.generation.default_resolution = int(os.getenv('FLOWCRAFT_DEFAULT_RESOLUTION', str(config.generation.default_resolution)))
        
        # Storage configuration
        config.storage.output_dir = os.getenv('FLOWCRAFT_OUTPUT_DIR', config.storage.output_dir)
        
        # Logging configuration
        config.logging.level = os.getenv('FLOWCRAFT_LOG_LEVEL', config.logging.level)
        config.logging.file = os.getenv('FLOWCRAFT_LOG_FILE', config.logging.file)
        
        return config
    
    def _validate_config(self, config: ProductionConfig):
        """Validate configuration values."""
        # Validate device
        valid_devices = ['cuda', 'cpu', 'mps', 'auto']
        if config.model.device not in valid_devices:
            raise ValueError(f"Invalid device: {config.model.device}. Must be one of {valid_devices}")
        
        # Validate dtype
        valid_dtypes = ['bfloat16', 'float16', 'float32']
        if config.model.dtype not in valid_dtypes:
            raise ValueError(f"Invalid dtype: {config.model.dtype}. Must be one of {valid_dtypes}")
        
        # Validate ports
        if not (1 <= config.api.port <= 65535):
            raise ValueError(f"Invalid API port: {config.api.port}")
        if not (1 <= config.ui.port <= 65535):
            raise ValueError(f"Invalid UI port: {config.ui.port}")
        
        # Validate generation parameters
        if not (1 <= config.generation.default_steps <= config.generation.max_steps):
            raise ValueError(f"Invalid default_steps: {config.generation.default_steps}")
        if not (1.0 <= config.generation.default_cfg_scale <= config.generation.max_cfg_scale):
            raise ValueError(f"Invalid default_cfg_scale: {config.generation.default_cfg_scale}")
        if not (config.generation.min_resolution <= config.generation.default_resolution <= config.generation.max_resolution):
            raise ValueError(f"Invalid default_resolution: {config.generation.default_resolution}")
        
        # Validate logging level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.logging.level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {config.logging.level}")
        
        logging.info("Configuration validated successfully")
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        log_level = getattr(logging, self.config.logging.level.upper())
        
        # Create logs directory if it doesn't exist
        log_file = Path(self.config.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=self.config.logging.format,
            handlers=[
                logging.FileHandler(
                    self.config.logging.file,
                    maxBytes=self.config.logging.max_bytes,
                    backupCount=self.config.logging.backup_count
                ),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"Logging configured at {self.config.logging.level} level")
    
    def setup_directories(self):
        """Setup required directories."""
        # Create output directories
        Path(self.config.storage.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.storage.api_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.storage.ui_output_dir).mkdir(parents=True, exist_ok=True)
        
        logging.info("Output directories created")


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ProductionConfig:
    """Get the global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager.config


def setup_config(config_path: Optional[str] = None) -> ConfigManager:
    """Setup configuration with logging and directories."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    _config_manager.setup_logging()
    _config_manager.setup_directories()
    return _config_manager
