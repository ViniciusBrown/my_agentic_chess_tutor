"""
Utility functions for chess operations.
"""
import chess
from typing import Dict, List, Tuple, Optional


def get_piece_counts(board: chess.Board) -> Dict[str, Dict[str, int]]:
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


def get_material_balance(board: chess.Board) -> int:
    """
    Calculate the material balance of the position in centipawns.
    
    Args:
        board (chess.Board): Chess board.
        
    Returns:
        int: Material balance in centipawns (positive for white advantage).
    """
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0  # King has no material value
    }
    
    balance = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                balance += value
            else:
                balance -= value
    
    return balance


def get_square_control(board: chess.Board) -> Dict[str, List[str]]:
    """
    Get all squares controlled by each side.
    
    Args:
        board (chess.Board): Chess board.
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping "white" and "black" to lists of controlled squares.
    """
    control = {
        "white": [],
        "black": []
    }
    
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
            if attacked_square_name not in control[color]:
                control[color].append(attacked_square_name)
    
    return control


def fen_to_board(fen: str) -> Optional[chess.Board]:
    """
    Convert a FEN string to a chess.Board object.
    
    Args:
        fen (str): FEN notation of the position.
        
    Returns:
        Optional[chess.Board]: Chess board, or None if the FEN is invalid.
    """
    try:
        return chess.Board(fen)
    except ValueError:
        return None


def is_valid_fen(fen: str) -> bool:
    """
    Check if a FEN string is valid.
    
    Args:
        fen (str): FEN notation of the position.
        
    Returns:
        bool: True if the FEN is valid, False otherwise.
    """
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False
