"""
Configuration module for Stockfish engine.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any


class StockfishConfig(BaseModel):
    """Configuration for Stockfish engine."""

    # Pydantic model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Analysis settings
    depth: int = Field(20, description="Analysis depth for Stockfish.")
    threads: int = Field(4, description="Number of CPU threads to use.")
    hash_size: int = Field(128, description="Hash table size in MB.")
    multi_pv: int = Field(3, description="Number of principal variations to calculate.")
    skill_level: int = Field(20, description="Stockfish skill level (0-20).")

    # Time control settings
    move_time: Optional[int] = Field(None, description="Maximum time to spend on a move in milliseconds.")

    # Binary settings
    binary_path: Optional[str] = Field(None, description="Path to Stockfish binary. If None, will use default.")

    # Cache settings
    cache_size: int = Field(1000, description="Maximum number of positions to keep in the analysis cache.")

    # Additional UCI options
    uci_options: Optional[Dict[str, Any]] = Field(None, description="Additional UCI options to set.")
