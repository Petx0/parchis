# Parchís Game Visualization

This module provides visualization capabilities for Parchís games, allowing you to replay games from log files with an animated board display.

## Features

- **Board Visualization**: Displays a complete Parchís board with all 68 main track squares, 4 player bases, and home columns
- **Game Replay**: Load any game log and replay it turn-by-turn
- **Interactive Mode**: Step through each turn with user input
- **Auto-Play Mode**: Watch the entire game replay automatically
- **Save Frames**: Export each turn as an image file for creating videos or animations
- **Piece Tracking**: Visual representation of all pieces with color coding
- **Move Highlighting**: Shows movement paths with arrows

## Installation Requirements

The visualizer requires matplotlib:

```bash
pip install matplotlib
```

## Usage

### Quick Start

Visualize the most recent game:

```bash
python -m parchis.visualization.visualize_game --latest
```

### Visualize a Specific Game

```bash
python -m parchis.visualization.visualize_game logs/game_20260111_201645_BLUE.json
```

### Auto-Play Mode (No Waiting)

```bash
python -m parchis.visualization.visualize_game logs/game_20260111_201645_BLUE.json --auto
```

### Save Each Frame as Image

```bash
python -m parchis.visualization.visualize_game logs/game_20260111_201645_BLUE.json --save-frames
```

This will create files named `replay_frame_0001.png`, `replay_frame_0002.png`, etc.

### End-to-End Demo (Play a Game, Then Replay It)

```bash
python -m parchis.visualization.demo_visualization
```

Plays a short 2-player game, saves the log, then replays it — useful when
you don't already have a log file handy.

## Interactive Controls

When running in interactive mode (default):
- Press **ENTER** to advance to the next turn
- Type **q** and press ENTER to quit the replay

## Board Layout

The board itself is `docs/images/foto_parchis.png` — a real, already-numbered
Parchís board photo — rendered as the figure's background. The visualizer
only draws the pieces, base circles, and move-highlight arrow on top of it;
it no longer draws its own squares, position numbers, or corner circles
(the photo already shows all of that).

- **Player Bases**: the photo's own corner circles
  - Red: Top-left
  - Blue: Top-right
  - Green: Bottom-left
  - Yellow: Bottom-right

- **Main Track**: the photo's 68 numbered squares, a genuine Parchís cross
  — four arms reaching from a center hub to each edge, each arm a straight
  3-row x 8-column block (two main-track flank rows + one color's private
  home-lane row): the track runs the length of one flank from the hub out
  to the edge, turns onto the home-lane row (that color's
  `HOME_ENTRY_POINTS` square), then runs back from the edge to the hub
  along the other flank

- **Home Columns**: each color's 7 private home-lane squares (69-75) run
  along the middle row of that color's own arm; position 76 (the finish)
  is the same physical point — the hub — for all 4 colors

- **Pieces**: Colored circles representing each player's pieces
  - Pieces in base are shown in the corner circles, one per quadrant of
    the circle
  - Pieces on the board are shown at their current positions, centered
    when alone on a square; if a safe square holds 2 pieces
    (`Board.MAX_PIECES_PER_SQUARE`), both are drawn smaller and offset so
    they stay visible instead of fully overlapping

## Using the Visualizer Programmatically

```python
from parchis.visualization.visualizer import ParchisVisualizer, replay_game_from_log

# Method 1: Replay a complete game
replay_game_from_log('logs/game.json', step_by_step=True)

# Method 2: Create custom visualizations
viz = ParchisVisualizer()
viz.create_board()

# Define game state (positions for each player's 4 pieces)
game_state = {
    'YELLOW': [None, 10, 25, None],  # None = in base
    'BLUE': [None, None, 30, 45],
    'RED': [15, None, None, 70],
    'GREEN': [None, 50, None, None]
}

viz.draw_pieces(game_state)
viz.save('my_board.png')
```

## Board Position System

- Positions 1-68: Main circular track
  - Yellow starts at 5, enters home at 68
  - Blue starts at 22, enters home at 17
  - Red starts at 39, enters home at 34
  - Green starts at 56, enters home at 51

- Positions 69-76: Home columns (final 8 squares to finish)
  - Position 76 is the final winning position

## Examples

### Example 1: Watch Latest Game

```bash
python -m parchis.visualization.visualize_game --latest
```

### Example 2: Create Video Frames

```bash
# Generate all frames
python -m parchis.visualization.visualize_game logs/my_game.json --auto --save-frames

# Convert to video with ffmpeg (optional)
ffmpeg -framerate 2 -pattern_type glob -i 'replay_frame_*.png' \
       -c:v libx264 -pix_fmt yuv420p game_replay.mp4
```

### Example 3: Demo Board

```python
from parchis.visualization.visualizer import ParchisVisualizer

viz = ParchisVisualizer()
viz.create_board()

# Show an empty board
viz.show()
```

## Visualization Details

### Color Scheme
- Yellow: `#FFD700` (Gold)
- Blue: `#4169E1` (Royal Blue)
- Red: `#DC143C` (Crimson)
- Green: `#228B22` (Forest Green)

### Board Features
- Safe squares, starting squares, and position numbers: shown by the
  `docs/images/foto_parchis.png` background photo itself, not drawn by code
- Piece markers: Solid colored circles with black outlines
- Movement arrows: Show path of last move

## Troubleshooting

### "No display" Error
If running on a headless server, use the non-interactive backend:

```python
import matplotlib
matplotlib.use('Agg')
```

### Small Pieces
Adjust piece size in the visualizer code by modifying the radius parameter in `_draw_piece_on_board()`.

### Overlapping Pieces
A square can hold up to `Board.MAX_PIECES_PER_SQUARE` (2) pieces at once
(own or opponent's, on a safe square). When it does, both are drawn
smaller and offset diagonally instead of centered — this is normal
behavior, not a bug.

## Future Enhancements

Potential improvements for future versions:
- More breathing room between the two circles when a square holds 2
  pieces (`ParchisVisualizer.SHARED_PIECE_RADIUS`/`SHARED_PIECE_OFFSET`)
- Animation between moves (smooth transitions)
- Highlight captured pieces
- Show dice roll results on screen
- Display game statistics during replay
- Support for custom board themes
- Interactive piece selection (click to see piece history)

## Technical Details

The visualizer uses:
- **matplotlib** for rendering, with `docs/images/foto_parchis.png` drawn via
  `imshow` as the board's background
- **patches** for drawing pieces and the move-highlight arrow
- **JSON** for loading game logs

Board coordinates are in that photo's own pixel space (631x622), not an
abstract grid — `y` is flipped once (`_pixel_to_data`) to match
matplotlib's bottom-up data coordinates, everything else is the photo's
raw pixels. `_calculate_position_coords()` builds one arm's coordinate
offsets by directly measuring the reference photo (`ARM_CELL_PITCH`,
`HUB_PIXEL`, etc. in `visualizer.py`) and rotates that template 90° three
times to cover all four colors — see `ARM_COLOR_BY_ROTATION_STEP` for
which color owns which arm. See `parchis/tests/test_visualizer.py` for
the geometry regression tests (bounds, distinct home lanes, no overlap
with a corner base circle, etc.); if the reference photo is ever replaced,
every pixel-measured constant needs recalibrating against the new file.

### Replay Performance

`replay_game_from_log` creates the board `Figure`/`Axes` once and reuses it
for every move (drawing/removing piece and highlight-arrow artists in
place) rather than recreating the whole figure per move — a long replay
with an interactive backend used to open one new window per move, which is
now fixed.
