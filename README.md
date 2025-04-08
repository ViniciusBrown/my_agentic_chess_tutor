# Chess Tutor AI

An intelligent chess teaching system that uses multiple specialized agents to analyze chess positions and provide instruction.

## Core Engine Integration

The first component of the Chess Tutor AI system is the core engine integration, which provides a high-level interface for interacting with the Stockfish chess engine using the `stockfish` and `python-chess` libraries.

### StockfishManager

The `StockfishManager` class is responsible for managing Stockfish engine instances and analysis requests. It provides the following features:

- Simple integration with the Stockfish chess engine
- Comprehensive chess position analysis
- Caching of analysis results for improved performance
- Support for advanced analysis features

### Installation

1. Make sure you have Stockfish installed on your system. You can download it from [the official website](https://stockfishchess.org/download/).
2. Clone this repository:
   ```
   git clone https://github.com/ViniciusBrown/my_agentic_chess_tutor.git
   cd my_agentic_chess_tutor
   ```
3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

Here's a simple example of how to use the `StockfishManager`:

```python
from chess_tutor.core.engine import StockfishConfig, StockfishManager

# Create a configuration
config = StockfishConfig(
    depth=20,
    threads=4,
    hash_size=128,
    multi_pv=3,
    skill_level=20
)

# Create and initialize the manager
manager = StockfishManager(config)

# Analyze a position
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position
analysis = manager.get_analysis(fen)

if analysis:
    print(f"Best move: {analysis.best_move}")
    print(f"Evaluation: {analysis.evaluation}")
    print(f"Depth: {analysis.depth_reached}")

    print("\nPrincipal variations:")
    for i, pv in enumerate(analysis.principal_variations):
        if pv.get('mate') is not None:
            score_str = f"Mate in {pv['mate']}"
        else:
            score_str = f"CP: {pv['centipawn']/100.0}"

        print(f"  {i+1}. {score_str} - Move: {pv['move']}")

# For advanced analysis using python-chess
from chess_tutor.core.engine import AdvancedAnalysis

advanced = AdvancedAnalysis()
detailed_analysis = advanced.analyze_position(fen, depth=20, multipv=3)

print(f"Best move: {detailed_analysis['best_move']}")
print(f"Score: {detailed_analysis['score']}")

for i, var in enumerate(detailed_analysis['variations']):
    print(f"Variation {i+1}: {' '.join(var['moves'][:5])}...")
```

### Running Tests

To run the tests:

```
python -m pytest tests
```

## Project Structure

```
chess_tutor/
├── core/
│   ├── engine/
│   │   ├── config.py         # Engine configuration
│   │   ├── stockfish_api.py  # Stockfish Python package integration
│   │   ├── chess_api.py      # Python-chess integration
│   │   ├── manager.py        # StockfishManager implementation
│   │   ├── analysis.py       # Analysis pipeline
│   │   └── __init__.py
│   └── __init__.py
├── utils/
│   ├── chess_utils.py      # Chess utility functions
│   └── __init__.py
├── requirements.txt      # Including stockfish and python-chess
└── __init__.py
```

## License

[MIT License](LICENSE)
