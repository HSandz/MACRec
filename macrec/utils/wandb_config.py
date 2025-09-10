import os
import wandb
from typing import Optional, Dict, Any
from loguru import logger

class WandbConfig:
    """Simple wandb configuration utility for MACRec project."""
    
    def __init__(self, project_name: str = "macrec", entity: Optional[str] = None):
        self.project_name = project_name
        self.entity = entity
        self.run = None
        
    def init(self, config: Optional[Dict[str, Any]] = None, tags: Optional[list] = None):
        """Initialize wandb run with simple configuration."""
        if self.run is not None:
            logger.warning("Wandb run already initialized")
            return
            
        # Set default config
        default_config = {
            "framework": "macrec",
            "version": "1.0"
        }
        
        if config:
            default_config.update(config)
            
        # Initialize wandb
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=default_config,
            tags=tags or []
        )
        
        logger.info(f"Wandb initialized: {self.run.url}")
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.run is None:
            logger.warning("Wandb not initialized, skipping log")
            return
            
        self.run.log(metrics, step=step)
        
    def log_model_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log model evaluation metrics with optional prefix."""
        if self.run is None:
            return
            
        prefixed_metrics = {}
        for key, value in metrics.items():
            prefixed_key = f"{prefix}_{key}" if prefix else key
            prefixed_metrics[prefixed_key] = value
            
        self.run.log(prefixed_metrics)
        
    def finish(self):
        """Finish the wandb run."""
        if self.run is not None:
            self.run.finish()
            self.run = None
            logger.info("Wandb run finished")
