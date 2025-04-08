"""
Python-chess integration for advanced chess analysis.
"""
import logging
import chess
import chess.engine
from typing import Dict, List, Optional, Any

from .config import StockfishConfig

logger = logging.getLogger(__name__)


class ChessAPI:
    """
    Wrapper for the python-chess library.
    Provides advanced chess analysis capabilities.
    """

    def __init__(self, config: StockfishConfig):
        """
        Initialize the ChessAPI.

        Args:
            config (StockfishConfig): Configuration for the Stockfish engine.
        """
        self.config = config
        self.stockfish_path = config.binary_path or "stockfish"

    def analyze_position(
        self,
        fen: str,
        depth: Optional[int] = None,
        multipv: Optional[int] = None,
        time_limit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze a chess position using python-chess and Stockfish.

        Args:
            fen (str): FEN notation of the position.
            depth (Optional[int]): Search depth. If None, uses the configured depth.
            multipv (Optional[int]): Number of principal variations. If None, uses the configured multi_pv.
            time_limit (Optional[float]): Time limit in seconds. If provided, overrides depth.

        Returns:
            Dict[str, Any]: Analysis results.
        """
        # Set default values from config
        depth = depth or self.config.depth
        multipv = multipv or self.config.multi_pv

        try:
            # Create a board from the FEN
            board = chess.Board(fen)

            # Start Stockfish process - use synchronous version
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

            try:
                # Configure engine
                if self.config.uci_options:
                    for name, value in self.config.uci_options.items():
                        engine.configure({name: value})

                # Set up analysis limits
                if time_limit:
                    limit = chess.engine.Limit(time=time_limit)
                else:
                    limit = chess.engine.Limit(depth=depth)

                # Run analysis
                result = engine.analyse(
                    board,
                    limit,
                    multipv=multipv
                )

                # Process and return results
                return self._process_analysis_result(result, board)

            finally:
                # Clean up resources
                engine.quit()

        except Exception as e:
            logger.error(f"Error analyzing position: {e}")
            return {
                "error": str(e),
                "best_move": None,
                "score": {"type": "cp", "value": 0},
                "variations": []
            }

    def _process_analysis_result(self, result: List[Dict[str, Any]], board) -> Dict[str, Any]:
        """
        Process the analysis result from python-chess.

        Args:
            result (List[Dict[str, Any]]): Analysis result from python-chess.
            board (chess.Board): The board that was analyzed.

        Returns:
            Dict[str, Any]: Processed analysis result.
        """
        # Initialize the result dictionary
        analysis = {
            "best_move": None,
            "score": None,
            "variations": [],
            "depth": 0
        }

        # Process each principal variation
        for i, pv_info in enumerate(result):
            # Extract score
            score = self._parse_score(pv_info["score"])

            # Extract moves
            moves = []
            if "pv" in pv_info:
                moves = [move.uci() for move in pv_info["pv"]]

            # Extract depth
            depth = pv_info.get("depth", 0)
            analysis["depth"] = max(analysis["depth"], depth)

            # Add variation to the list
            variation = {
                "moves": moves,
                "score": score,
                "depth": depth
            }
            analysis["variations"].append(variation)

            # Set best move and score from the first (best) variation
            if i == 0:
                analysis["best_move"] = moves[0] if moves else None
                analysis["score"] = score

        return analysis

    def _parse_score(self, score) -> Dict[str, Any]:
        """
        Parse a score object from python-chess.

        Args:
            score: Score object from python-chess.

        Returns:
            Dict[str, Any]: Parsed score.
        """
        # Handle PovScore objects
        if hasattr(score, 'relative'):
            score = score.relative

        # Check if it's a mate score
        if hasattr(score, 'mate') and score.mate() is not None:
            return {"type": "mate", "value": score.mate()}
        else:
            # Handle centipawn score
            try:
                return {"type": "cp", "value": score.score()}
            except AttributeError:
                # Fallback for numeric scores
                try:
                    return {"type": "cp", "value": int(score)}
                except (TypeError, ValueError):
                    return {"type": "cp", "value": 0}

    def get_legal_moves(self, fen: str) -> List[Dict[str, Any]]:
        """
        Get all legal moves for a position with additional information.

        Args:
            fen (str): FEN notation of the position.

        Returns:
            List[Dict[str, Any]]: List of legal moves with additional information.
        """
        try:
            # Create a board from the FEN
            board = chess.Board(fen)

            # Get all legal moves
            legal_moves = []
            for move in board.legal_moves:
                # Get move information
                move_info = {
                    "uci": move.uci(),
                    "san": board.san(move),
                    "from_square": chess.square_name(move.from_square),
                    "to_square": chess.square_name(move.to_square),
                    "promotion": chess.piece_symbol(move.promotion) if move.promotion else None,
                    "is_capture": board.is_capture(move),
                    "is_check": board.gives_check(move),
                    "piece_moved": board.piece_at(move.from_square).symbol() if board.piece_at(move.from_square) else None
                }

                # Make the move to check if it's checkmate or stalemate
                board.push(move)
                move_info["is_checkmate"] = board.is_checkmate()
                move_info["is_stalemate"] = board.is_stalemate()
                board.pop()

                legal_moves.append(move_info)

            return legal_moves

        except Exception as e:
            logger.error(f"Error getting legal moves: {e}")
            return []

    def get_position_features(self, fen: str) -> Dict[str, Any]:
        """
        Extract various features from a chess position.

        Args:
            fen (str): FEN notation of the position.

        Returns:
            Dict[str, Any]: Dictionary of position features.
        """
        try:
            # Create a board from the FEN
            board = chess.Board(fen)

            # Initialize features dictionary
            features = {
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "fullmove_number": board.fullmove_number,
                "halfmove_clock": board.halfmove_clock,
                "can_castle_kingside": {
                    "white": board.has_kingside_castling_rights(chess.WHITE),
                    "black": board.has_kingside_castling_rights(chess.BLACK)
                },
                "can_castle_queenside": {
                    "white": board.has_queenside_castling_rights(chess.WHITE),
                    "black": board.has_queenside_castling_rights(chess.BLACK)
                },
                "is_check": board.is_check(),
                "is_checkmate": board.is_checkmate(),
                "is_stalemate": board.is_stalemate(),
                "is_insufficient_material": board.is_insufficient_material(),
                "is_game_over": board.is_game_over(),
                "piece_counts": self._get_piece_counts(board),
                "piece_locations": self._get_piece_locations(board),
                "control_counts": self._get_square_control_counts(board),
                "pawn_structure": self._analyze_pawn_structure(board)
            }

            return features

        except Exception as e:
            logger.error(f"Error getting position features: {e}")
            return {"error": str(e)}

    def _get_piece_counts(self, board: chess.Board) -> Dict[str, Dict[str, int]]:
        """
        Count the pieces on the board by type and color.

        Args:
            board (chess.Board): Chess board.

        Returns:
            Dict[str, Dict[str, int]]: Piece counts by type and color.
        """
        counts = {
            "white": {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 0, "K": 0},
            "black": {"p": 0, "n": 0, "b": 0, "r": 0, "q": 0, "k": 0}
        }

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color = "white" if piece.color == chess.WHITE else "black"
                counts[color][piece.symbol()] += 1

        return counts

    def _get_piece_locations(self, board: chess.Board) -> Dict[str, List[str]]:
        """
        Get the locations of all pieces on the board.

        Args:
            board (chess.Board): Chess board.

        Returns:
            Dict[str, List[str]]: Piece locations by piece symbol.
        """
        locations = {}

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                symbol = piece.symbol()
                if symbol not in locations:
                    locations[symbol] = []
                locations[symbol].append(chess.square_name(square))

        return locations

    def _get_square_control_counts(self, board: chess.Board) -> Dict[str, Dict[str, int]]:
        """
        Count how many times each square is controlled by each side.

        Args:
            board (chess.Board): Chess board.

        Returns:
            Dict[str, Dict[str, int]]: Square control counts by color.
        """
        control = {
            "white": {},
            "black": {}
        }

        # Initialize all squares with 0 control
        for square_name in [chess.square_name(sq) for sq in chess.SQUARES]:
            control["white"][square_name] = 0
            control["black"][square_name] = 0

        # Check control for each piece
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue

            # Get the color
            color = "white" if piece.color == chess.WHITE else "black"

            # Get all squares attacked by this piece
            for attacked_square in board.attacks(square):
                attacked_square_name = chess.square_name(attacked_square)
                control[color][attacked_square_name] += 1

        return control

    def _analyze_pawn_structure(self, board: chess.Board) -> Dict[str, Any]:
        """
        Analyze the pawn structure of the position.

        Args:
            board (chess.Board): Chess board.

        Returns:
            Dict[str, Any]: Pawn structure analysis.
        """
        # Initialize pawn structure analysis
        pawn_structure = {
            "isolated_pawns": {"white": [], "black": []},
            "doubled_pawns": {"white": [], "black": []},
            "backward_pawns": {"white": [], "black": []},
            "passed_pawns": {"white": [], "black": []},
            "pawn_islands": {"white": 0, "black": 0}
        }

        # Get pawn locations by file
        white_pawns_by_file = [[] for _ in range(8)]
        black_pawns_by_file = [[] for _ in range(8)]

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue

            if piece.piece_type == chess.PAWN:
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                square_name = chess.square_name(square)

                if piece.color == chess.WHITE:
                    white_pawns_by_file[file_idx].append((rank_idx, square_name))
                else:
                    black_pawns_by_file[file_idx].append((rank_idx, square_name))

        # Sort pawns by rank
        for file_idx in range(8):
            white_pawns_by_file[file_idx].sort()
            black_pawns_by_file[file_idx].sort(reverse=True)

        # Analyze isolated pawns
        for file_idx in range(8):
            # White isolated pawns
            if white_pawns_by_file[file_idx] and (
                (file_idx == 0 or not white_pawns_by_file[file_idx - 1]) and
                (file_idx == 7 or not white_pawns_by_file[file_idx + 1])
            ):
                for _, square_name in white_pawns_by_file[file_idx]:
                    pawn_structure["isolated_pawns"]["white"].append(square_name)

            # Black isolated pawns
            if black_pawns_by_file[file_idx] and (
                (file_idx == 0 or not black_pawns_by_file[file_idx - 1]) and
                (file_idx == 7 or not black_pawns_by_file[file_idx + 1])
            ):
                for _, square_name in black_pawns_by_file[file_idx]:
                    pawn_structure["isolated_pawns"]["black"].append(square_name)

        # Analyze doubled pawns
        for file_idx in range(8):
            # White doubled pawns
            if len(white_pawns_by_file[file_idx]) > 1:
                for _, square_name in white_pawns_by_file[file_idx]:
                    pawn_structure["doubled_pawns"]["white"].append(square_name)

            # Black doubled pawns
            if len(black_pawns_by_file[file_idx]) > 1:
                for _, square_name in black_pawns_by_file[file_idx]:
                    pawn_structure["doubled_pawns"]["black"].append(square_name)

        # Count pawn islands
        white_islands = 0
        black_islands = 0

        for file_idx in range(8):
            # White islands
            if white_pawns_by_file[file_idx] and (file_idx == 0 or not white_pawns_by_file[file_idx - 1]):
                white_islands += 1

            # Black islands
            if black_pawns_by_file[file_idx] and (file_idx == 0 or not black_pawns_by_file[file_idx - 1]):
                black_islands += 1

        pawn_structure["pawn_islands"]["white"] = white_islands
        pawn_structure["pawn_islands"]["black"] = black_islands

        # Analyze passed pawns
        for file_idx in range(8):
            # White passed pawns
            for rank_idx, square_name in white_pawns_by_file[file_idx]:
                is_passed = True

                # Check if there are any black pawns that can block or capture
                for f in range(max(0, file_idx - 1), min(8, file_idx + 2)):
                    for r, _ in black_pawns_by_file[f]:
                        if r > rank_idx:
                            is_passed = False
                            break
                    if not is_passed:
                        break

                if is_passed:
                    pawn_structure["passed_pawns"]["white"].append(square_name)

            # Black passed pawns
            for rank_idx, square_name in black_pawns_by_file[file_idx]:
                is_passed = True

                # Check if there are any white pawns that can block or capture
                for f in range(max(0, file_idx - 1), min(8, file_idx + 2)):
                    for r, _ in white_pawns_by_file[f]:
                        if r < rank_idx:
                            is_passed = False
                            break
                    if not is_passed:
                        break

                if is_passed:
                    pawn_structure["passed_pawns"]["black"].append(square_name)

        return pawn_structure
