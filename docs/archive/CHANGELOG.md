# Changelog

## Version 1.1 - Rules Correction Update (2026-01-10)

### Major Rule Changes

This update corrects the game rules to match the proper Parchís board layout and mechanics.

#### Board Changes

**Circular Track:**
- Circular track now has 68 squares numbered 1-68 (previously 0-67)
- Track wraps around: after square 68 comes square 1

**Starting Positions (Player-Specific):**
- Yellow starts at square 5 (previously all players at 0)
- Blue starts at square 22 (previously all at 0)
- Red starts at square 39 (previously all at 0)
- Green starts at square 56 (previously all at 0)

**Home Entry Points (Player-Specific):**
- Yellow enters home after square 68
- Blue enters home after square 17
- Red enters home after square 34
- Green enters home after square 51

**Home Columns:**
- Each player has 8 squares (69-76) in their home column (previously 7 squares, 68-74)
- Final position is now square 76 (previously 74)

**Safe Squares:**
- Starting squares: 5, 22, 39, 56 (one per player)
- Additional safe squares: 12, 17, 29, 34, 46, 51, 63, 68
- Previously: 0, 5, 12, 17, 22, 29, 34, 39, 46, 51, 56, 63

**Player Order:**
- Changed from RED → BLUE → YELLOW → GREEN
- Now: YELLOW → BLUE → RED → GREEN

### Implementation Changes

#### Files Modified

1. **[RULES.md](RULES.md)**
   - Complete rewrite of board layout section
   - Updated all position references
   - Added detailed path examples for each player

2. **[parchis/board.py](parchis/board.py)**
   - Updated constants: `MAIN_TRACK_SIZE`, `HOME_COLUMN_SIZE`, `FINAL_POSITION`
   - New constants: `STARTING_POSITIONS`, `HOME_ENTRY_POINTS`, `HOME_COLUMN_START`
   - Updated `SAFE_SQUARES` set
   - Updated all docstring position ranges (1-76 instead of 0-74)

3. **[parchis/player.py](parchis/player.py)**
   - Changed `COLORS` order to `["YELLOW", "BLUE", "RED", "GREEN"]`
   - Added `starting_position` and `home_entry_point` attributes
   - Players now initialize with color-specific starting positions
   - Updated constructor to accept optional `starting_position` parameter

4. **[parchis/piece.py](parchis/piece.py)**
   - Updated `mark_finished()` to set position to 76 (was 74)
   - Updated docstrings for position ranges

5. **[parchis/game.py](parchis/game.py)**
   - Complete rewrite of `get_legal_moves()` method
   - Added `_calculate_new_position()` helper method for circular track with wrapping
   - Handles player-specific starting positions for piece entry
   - Implements proper home entry logic based on player color
   - Supports pieces moving in home column (69-76)

6. **[test_game.py](test_game.py)**
   - Updated all tests to use new position numbers
   - Added tests for player-specific starting positions
   - Added tests for home entry points
   - Verified final position is 76
   - Changed test player colors to YELLOW and BLUE

### Key Algorithm Changes

#### Movement Calculation
The new `_calculate_new_position()` method:
- Simulates movement step-by-step
- Handles wrapping from square 68 to square 1
- Detects when a piece reaches its home entry point
- Transitions pieces from main track to home column
- Returns appropriate position (main track or home column)

#### Home Entry Logic
- Each player has a specific square where they enter their home column
- When a piece lands on or passes through the home entry point, it enters the home column at square 69
- If movement continues beyond entry point, piece moves further into home column (70, 71, etc.)

### Testing

All tests pass successfully:
- ✓ Safe squares correctly identified
- ✓ Player-specific starting positions (5, 22, 39, 56)
- ✓ Player-specific home entry points (68, 17, 34, 51)
- ✓ Home column movement (69-76)
- ✓ Final position at 76
- ✓ Complete games finish successfully
- ✓ Game logging includes all new positions

### Backward Compatibility

**Breaking Changes:**
This is a **breaking change** from version 1.0. Game logs and saved games from version 1.0 are not compatible with version 1.1 due to:
- Different position numbering scheme
- Player-specific starting positions
- Different final position (76 vs 74)
- Changed player order

### Migration Notes

If you have existing game logs from version 1.0:
- They will show positions 0-74 (old system)
- All players started at position 0 (now they start at 5, 22, 39, or 56)
- Games finished at position 74 (now 76)

The old logs are still valid representations of version 1.0 games, but cannot be replayed in version 1.1.
