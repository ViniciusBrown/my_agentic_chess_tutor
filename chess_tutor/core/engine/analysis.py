"""
Analysis pipeline for processing chess positions.
"""
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, ConfigDict
from .manager import StockfishManager, EngineAnalysis

logger = logging.getLogger(__name__)


class PositionAnalysisRequest(BaseModel):
    """Request for position analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fen: str
    depth: Optional[int] = None
    multipv: Optional[int] = None
    include_features: bool = False
    include_legal_moves: bool = False
    include_advanced_analysis: bool = False


class PositionAnalysisResult(BaseModel):
    """Result of position analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fen: str
    basic_analysis: EngineAnalysis
    advanced_analysis: Optional[Dict[str, Any]] = None
    position_features: Optional[Dict[str, Any]] = None
    legal_moves: Optional[List[Dict[str, Any]]] = None


class AnalysisPipeline:
    """
    Coordinates analysis requests for chess positions.
    Provides batch processing and result aggregation.
    """

    def __init__(self, manager: StockfishManager):
        """
        Initialize the AnalysisPipeline.

        Args:
            manager (StockfishManager): StockfishManager instance.
        """
        self.manager = manager
        self.max_workers = 4  # Number of concurrent analysis threads

    def analyze_position(self, request: PositionAnalysisRequest) -> PositionAnalysisResult:
        """
        Analyze a single chess position.

        Args:
            request (PositionAnalysisRequest): Analysis request.

        Returns:
            PositionAnalysisResult: Analysis result.
        """
        # Get basic analysis
        basic_analysis = self.manager.get_analysis(
            fen=request.fen,
            depth=request.depth
        )

        # Initialize result
        result = PositionAnalysisResult(
            fen=request.fen,
            basic_analysis=basic_analysis
        )

        # Get advanced analysis if requested
        if request.include_advanced_analysis:
            result.advanced_analysis = self.manager.get_advanced_analysis(
                fen=request.fen,
                depth=request.depth,
                multipv=request.multipv
            )

        # Get position features if requested
        if request.include_features:
            result.position_features = self.manager.get_position_features(request.fen)

        # Get legal moves if requested
        if request.include_legal_moves:
            result.legal_moves = self.manager.get_legal_moves(request.fen)

        return result

    def analyze_positions(self, requests: List[PositionAnalysisRequest]) -> List[PositionAnalysisResult]:
        """
        Analyze multiple chess positions concurrently.

        Args:
            requests (List[PositionAnalysisRequest]): List of analysis requests.

        Returns:
            List[PositionAnalysisResult]: List of analysis results.
        """
        # Use ThreadPoolExecutor for concurrent analysis
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all analysis tasks
            futures = [executor.submit(self.analyze_position, request) for request in requests]

            # Collect results
            results = [future.result() for future in futures]

        return results

    def analyze_game(self, fen_positions: List[str], depth: Optional[int] = None) -> Dict[str, EngineAnalysis]:
        """
        Analyze all positions in a game.

        Args:
            fen_positions (List[str]): List of FEN positions from a game.
            depth (Optional[int]): Analysis depth. If None, uses default.

        Returns:
            Dict[str, EngineAnalysis]: Dictionary mapping FEN strings to analysis results.
        """
        return self.manager.analyze_multiple_positions(fen_positions, depth)

    def annotate_moves(self, fen_positions: List[str], moves: List[str], depth: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Annotate a sequence of moves with analysis.

        Args:
            fen_positions (List[str]): List of FEN positions.
            moves (List[str]): List of moves in UCI format.
            depth (Optional[int]): Analysis depth. If None, uses default.

        Returns:
            List[Dict[str, Any]]: List of annotated moves.
        """
        # Analyze all positions
        analysis_results = self.analyze_game(fen_positions, depth)

        # Create annotated moves
        annotated_moves = []

        for i, (fen, move) in enumerate(zip(fen_positions, moves)):
            # Get analysis for this position
            analysis = analysis_results.get(fen)

            if not analysis:
                continue

            # Create annotation
            annotation = {
                "move_number": i + 1,
                "fen": fen,
                "move": move,
                "evaluation": analysis.evaluation,
                "best_move": analysis.best_move,
                "is_best": move == analysis.best_move,
                "alternatives": [
                    {
                        "move": pv["move"],
                        "evaluation": pv["centipawn"] / 100.0 if "centipawn" in pv else float("inf") if pv.get("mate") else 0.0
                    }
                    for pv in analysis.principal_variations[:3]  # Top 3 alternatives
                ]
            }

            annotated_moves.append(annotation)

        return annotated_moves
