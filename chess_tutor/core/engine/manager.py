"""
StockfishManager for managing Stockfish engine instances and analysis requests.
"""
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict

from .config import StockfishConfig
from .stockfish_api import StockfishAPI
from .chess_api import ChessAPI

logger = logging.getLogger(__name__)


class EngineAnalysis(BaseModel):
    """Structured engine analysis results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    evaluation: float
    best_move: str
    principal_variations: List[Dict[str, Any]]
    depth_reached: int
    nodes_searched: int = 0
    time_spent_ms: int = 0


class StockfishManager:
    """
    Manages Stockfish engine instances and analysis requests.
    Provides a high-level interface for position analysis.
    """

    def __init__(self, config: Optional[StockfishConfig] = None):
        """
        Initialize the StockfishManager.

        Args:
            config (Optional[StockfishConfig]): Configuration for Stockfish.
                If None, default configuration will be used.
        """
        self.config = config or StockfishConfig()
        self.stockfish_api = StockfishAPI(self.config)
        self.chess_api = ChessAPI(self.config)
        self.analysis_cache: Dict[str, EngineAnalysis] = {}

    def get_analysis(
        self,
        fen: str,
        depth: Optional[int] = None,
        use_cache: bool = True
    ) -> EngineAnalysis:
        """
        Get analysis for a chess position.

        Args:
            fen (str): FEN notation of the position.
            depth (Optional[int]): Search depth. If None, uses config default.
            use_cache (bool): Whether to use the analysis cache.

        Returns:
            EngineAnalysis: Analysis results.
        """
        # Set default depth from config
        depth = depth or self.config.depth

        # Create cache key
        cache_key = self._create_cache_key(fen, depth)

        # Check cache
        if use_cache and cache_key in self.analysis_cache:
            logger.debug(f"Cache hit for position {fen[:20]}...")
            return self.analysis_cache[cache_key]

        try:
            # Set position
            self.stockfish_api.set_position(fen)

            # Get evaluation
            evaluation = self.stockfish_api.get_evaluation(depth)

            # Get best move
            best_move = self.stockfish_api.get_best_move(depth)

            # Get top moves
            top_moves = self.stockfish_api.get_top_moves(self.config.multi_pv)

            # Create analysis result
            analysis = EngineAnalysis(
                evaluation=evaluation["value"] / 100.0 if evaluation["type"] == "cp" else float("inf") if evaluation["type"] == "mate" else 0.0,
                best_move=best_move or "",
                principal_variations=[
                    {
                        "move": move["Move"],
                        "centipawn": move["Centipawn"],
                        "mate": move.get("Mate")
                    }
                    for move in top_moves
                ],
                depth_reached=depth,
                nodes_searched=0,  # Not directly available from stockfish module
                time_spent_ms=0    # Not directly available from stockfish module
            )

            # Cache the result
            if use_cache:
                self.analysis_cache[cache_key] = analysis

                # Limit cache size
                if len(self.analysis_cache) > self.config.cache_size:
                    self._prune_cache()

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing position: {e}")
            # Return a default analysis
            return EngineAnalysis(
                evaluation=0.0,
                best_move="",
                principal_variations=[],
                depth_reached=0,
                nodes_searched=0,
                time_spent_ms=0
            )

    def get_advanced_analysis(
        self,
        fen: str,
        depth: Optional[int] = None,
        multipv: Optional[int] = None,
        time_limit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get advanced analysis for a chess position using python-chess.

        Args:
            fen (str): FEN notation of the position.
            depth (Optional[int]): Search depth. If None, uses config default.
            multipv (Optional[int]): Number of principal variations. If None, uses config default.
            time_limit (Optional[float]): Time limit in seconds. If provided, overrides depth.

        Returns:
            Dict[str, Any]: Advanced analysis results.
        """
        return self.chess_api.analyze_position(
            fen=fen,
            depth=depth or self.config.depth,
            multipv=multipv or self.config.multi_pv,
            time_limit=time_limit
        )

    def get_position_features(self, fen: str) -> Dict[str, Any]:
        """
        Get various features of a chess position.

        Args:
            fen (str): FEN notation of the position.

        Returns:
            Dict[str, Any]: Position features.
        """
        return self.chess_api.get_position_features(fen)

    def get_legal_moves(self, fen: str) -> List[Dict[str, Any]]:
        """
        Get all legal moves for a position with additional information.

        Args:
            fen (str): FEN notation of the position.

        Returns:
            List[Dict[str, Any]]: List of legal moves with additional information.
        """
        return self.chess_api.get_legal_moves(fen)

    def analyze_multiple_positions(
        self,
        positions: List[str],
        depth: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict[str, EngineAnalysis]:
        """
        Analyze multiple positions.

        Args:
            positions (List[str]): List of FEN strings.
            depth (Optional[int]): Search depth. If None, uses config default.
            use_cache (bool): Whether to use the analysis cache.

        Returns:
            Dict[str, EngineAnalysis]: Dictionary mapping FEN strings to analysis results.
        """
        results = {}

        for fen in positions:
            results[fen] = self.get_analysis(fen, depth, use_cache)

        return results

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self.analysis_cache.clear()

    def _create_cache_key(self, fen: str, depth: int) -> str:
        """
        Create a cache key for a position and analysis parameters.

        Args:
            fen (str): FEN notation of the position.
            depth (int): Search depth.

        Returns:
            str: Cache key.
        """
        # Normalize FEN by removing extra spaces
        normalized_fen = ' '.join(fen.split())

        # Create key with parameters
        return f"{normalized_fen}|d{depth}"

    def _prune_cache(self) -> None:
        """
        Prune the cache to keep it within the size limit.
        Removes the oldest entries first.
        """
        # Simple implementation: remove random entries
        # In a real implementation, you might want to use LRU or another strategy
        excess = len(self.analysis_cache) - self.config.cache_size
        if excess <= 0:
            return

        keys_to_remove = list(self.analysis_cache.keys())[:excess]
        for key in keys_to_remove:
            del self.analysis_cache[key]

    @property
    def cache_size(self) -> int:
        """Get the current number of entries in the cache."""
        return len(self.analysis_cache)
