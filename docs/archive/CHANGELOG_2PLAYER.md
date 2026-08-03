# Changelog: 2-Player Opposite Colors

## Version 2.1 - Two-Player Balance Update

### Date: January 11, 2026

### Overview
Updated 2-player games to always use opposite colors for better game balance and strategic gameplay.

### Change Details

#### What Changed
In 2-player games, players now always play with opposite colors:
- **Red vs Yellow** (opposite corners on board)
- **Blue vs Green** (opposite corners on board)

The color pair is randomly selected at game start, providing variety while ensuring balanced gameplay.

#### Why This Change
Using opposite colors in 2-player games provides:
1. **Maximum Distance**: Players start on opposite sides of the board (34 squares apart)
2. **Strategic Balance**: Equal opportunities for captures and blockades
3. **Traditional Play**: Matches standard Parchís 2-player conventions
4. **Visual Clarity**: Easier to distinguish pieces when colors are maximally different

#### Previous Behavior
Before this change, 2-player games used the first two colors in order:
- Always Yellow vs Blue (adjacent starting positions, only 17 squares apart)
- Less strategic depth due to starting positions being too close

#### Implementation

**File Modified:** `parchis/game.py`

```python
# Create players with appropriate colors
# For 2-player games, use opposite colors (Red vs Yellow or Blue vs Green)
if num_players == 2:
    # Randomly choose between the two opposite color pairs
    import random
    if random.choice([True, False]):
        colors = ["RED", "YELLOW"]  # Opposite on board
    else:
        colors = ["BLUE", "GREEN"]  # Opposite on board
else:
    colors = Player.COLORS[:num_players]
```

**Lines Changed:** 29-39 in game.py

### Board Positions

Starting positions for each color pair:

**Red vs Yellow:**
- Red starts at position 39
- Yellow starts at position 5
- Distance: 34 squares (half the board)
- Red enters home at 34, Yellow enters home at 68

**Blue vs Green:**
- Blue starts at position 22
- Green starts at position 56
- Distance: 34 squares (half the board)
- Blue enters home at 17, Green enters home at 51

### Testing

Verified with 10 test games:
```
Game 1: YELLOW vs RED
Game 2: BLUE vs GREEN
Game 3: GREEN vs BLUE
Game 4: RED vs YELLOW
Game 5: GREEN vs BLUE
Game 6: YELLOW vs RED
Game 7: RED vs YELLOW
Game 8: GREEN vs BLUE
Game 9: BLUE vs GREEN
Game 10: GREEN vs BLUE

✓ All 2-player games use opposite colors!
```

All existing tests pass with this change:
- ✅ test_basic_game (2 players)
- ✅ All rule tests
- ✅ Bonus system tests
- ✅ Logger tests

### Documentation Updates

Updated documentation to reflect 2-player color rules:

**README.md:**
- Added note: "**2-player games**: Always use opposite colors (Red vs Yellow, or Blue vs Green)"

**RULES.md:**
- Added **Player Colors** section explaining color selection for 2, 3, and 4 player games

### Game Logs

Game logs now show the correct opposite colors in metadata:
```json
{
  "metadata": {
    "players": [
      {"id": 0, "color": "RED"},
      {"id": 1, "color": "YELLOW"}
    ]
  }
}
```

### Backward Compatibility

**Breaking Change:** Yes, but minimal impact
- Existing 2-player game logs will show YELLOW vs BLUE
- New 2-player games will show RED vs YELLOW or BLUE vs GREEN
- All game mechanics remain identical
- Log format unchanged

### Impact on Statistics

This change affects 2-player game analysis:
- Previous distribution tests used YELLOW vs BLUE
- Future tests will include all valid color pairs
- Statistical analysis remains valid (win distributions by turn order, not color)

### 3 and 4 Player Games

No changes to 3 and 4 player games:
- **3 players**: Yellow, Blue, Red (unchanged)
- **4 players**: Yellow, Blue, Red, Green (unchanged)

### Usage Examples

```bash
# Play a 2-player game (will use opposite colors)
python main.py --mode full --players 2

# Colors will be either:
# - Red vs Yellow, or
# - Blue vs Green
```

### Visual Reference

On the board visualization:
- Red (top-left corner) vs Yellow (bottom-right corner) = diagonal opposition
- Blue (top-right corner) vs Green (bottom-left corner) = diagonal opposition

Both configurations provide maximum starting distance and balanced gameplay.

### Future Considerations

Potential enhancements:
- [ ] Allow users to manually select color pair for 2-player games
- [ ] Add color preference to player settings
- [ ] Track statistics separately by color pair
- [ ] Analyze if one color pair has strategic advantage over the other

### Summary

This update ensures 2-player Parchís games are more balanced and strategic by using opposite colors, following traditional game conventions while maintaining all existing game mechanics and logging functionality.
