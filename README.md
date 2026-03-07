# Capture Go

A Python implementation of the classic board game Go, featuring an AI opponent with adjustable difficulty levels. Players compete on a 9x9 board, aiming to capture the most stones according to a simplified win condition.

## Features

*   Play against a computer opponent using a Monte Carlo Tree Search (MCTS) AI.
*   Adjustable AI difficulty: Choose from Easy, Medium, Hard, or Very Hard.
*   Visual game board with standard Go aesthetics.
*   Tracks captured stones for both players.
*   Win detection based on capturing a threshold number of stones (default: 15).
*   Simple and intuitive click-based controls for placing stones.

## Requirements

*   Python 3.x (Tested with Python 3.7+)
*   Libraries:
    *   `pygame` (for graphics and user interface)
    *   `numpy` (for board representation and calculations)

## Installation

### From Source

1.  Ensure you have Python installed.
2.  Install the required libraries using pip:
    ```bash
    pip install pygame numpy
    ```
3.  Download the `gogame.py` file.
4.  Run the game from the command line:
    ```bash
    python gogame.py
    ```

### From Executable (if available)

1.  Download the packaged executable folder (e.g., `Capture Go`).
2.  Navigate into the folder.
3.  Run the `Capture Go.exe` file.

## How to Play

*   The human player plays as Black and goes first.
*   Click on an empty intersection to place your stone.
*   The AI opponent (White) will automatically make a move after yours.
*   Try to surround and capture your opponent's stones.
*   The first player to capture the required number of stones (see Features) wins the game.
*   **Changing Difficulty:** During the game, press the following keys to adjust the AI's strength:
    *   `1`: Easy
    *   `2`: Medium
    *   `3`: Hard
    *   `4`: Very Hard
    The current difficulty level is displayed at the top of the window.

## Notes

*   This implementation uses a simplified win condition based on captures rather than standard territory scoring.
*   The AI's strength is primarily controlled by the number of MCTS iterations performed per move. Higher difficulties use more iterations, resulting in stronger but slower play.

## License

This project is open-source.
