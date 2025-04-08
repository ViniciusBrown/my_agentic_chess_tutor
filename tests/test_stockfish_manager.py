"""
Tests for the StockfishManager class using stockfish and python-chess.
"""
import unittest
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chess_tutor.core.engine import StockfishConfig, StockfishManager


class TestStockfishManager(unittest.TestCase):
    """Test cases for the StockfishManager class."""

    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def setUp(self):
        """Set up each test."""
        self.config = StockfishConfig(
            depth=10,
            threads=1,
            hash_size=16,
            multi_pv=1,
            skill_level=20
        )

        self.manager = StockfishManager(self.config)

    def test_basic_analysis(self):
        """Test that the manager can analyze a position."""
        # Starting position
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        analysis = self.manager.get_analysis(fen)

        self.assertIsNotNone(analysis)
        self.assertIsNotNone(analysis.best_move)
        self.assertGreaterEqual(len(analysis.principal_variations), 1)
        self.assertEqual(analysis.depth_reached, 10)

    def test_advanced_analysis(self):
        """Test that the manager can perform advanced analysis."""
        # Starting position
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        analysis = self.manager.get_advanced_analysis(fen, depth=10, multipv=3)

        self.assertIsNotNone(analysis)
        self.assertIsNotNone(analysis['best_move'])
        self.assertIsNotNone(analysis['score'])
        self.assertGreaterEqual(len(analysis['variations']), 1)

    def test_position_features(self):
        """Test that the manager can extract position features."""
        # Starting position
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        features = self.manager.get_position_features(fen)

        self.assertIsNotNone(features)
        self.assertEqual(features['side_to_move'], 'white')
        self.assertTrue(features['can_castle_kingside']['white'])
        self.assertTrue(features['can_castle_queenside']['white'])
        self.assertTrue(features['can_castle_kingside']['black'])
        self.assertTrue(features['can_castle_queenside']['black'])
        self.assertFalse(features['is_check'])
        self.assertFalse(features['is_checkmate'])
        self.assertFalse(features['is_stalemate'])

    def test_legal_moves(self):
        """Test that the manager can get legal moves."""
        # Starting position
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = self.manager.get_legal_moves(fen)

        self.assertIsNotNone(moves)
        self.assertEqual(len(moves), 20)  # 20 legal moves in the starting position

    def test_cache(self):
        """Test that the analysis cache works correctly."""
        # Starting position
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        # First analysis (should not be cached)
        analysis1 = self.manager.get_analysis(fen)
        self.assertIsNotNone(analysis1)

        # Second analysis (should be cached)
        analysis2 = self.manager.get_analysis(fen)
        self.assertIsNotNone(analysis2)

        # The results should be identical
        self.assertEqual(analysis1.best_move, analysis2.best_move)

        # Cache size should be 1
        self.assertEqual(self.manager.cache_size, 1)

        # Clear cache
        self.manager.clear_cache()
        self.assertEqual(self.manager.cache_size, 0)

    def test_multiple_positions(self):
        """Test that the manager can analyze multiple positions."""
        positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After 1. e4
        ]

        results = self.manager.analyze_multiple_positions(positions)

        self.assertEqual(len(results), 2)
        for fen, result in results.items():
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.best_move)


if __name__ == '__main__':
    unittest.main()
