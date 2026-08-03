# Changelog - Version 1.2

## Major Rule Corrections (2026-01-10)

This update implements the correct Parchís rules as specified by the user.

### Critical Rule Changes

#### 1. Entry Roll Changed from 6 to 5
- **Previous**: Players needed to roll a 6 to bring pieces from base
- **New**: Players need to roll a **5** to bring pieces from base
- Can capture opponent pieces at starting square when entering

#### 2. Bonus Turns for Rolling 6
- **Previous**: No bonus turns
- **New**: Rolling a 6 grants an additional roll
- Player continues rolling and moving until they roll something other than 6
- Bonus turns apply even if player had no legal moves

#### 3. Three Consecutive 6s Penalty
- **New Rule**: If a player rolls three 6s in a row during the same turn:
  - The piece that was moved with the **second** 6 is captured and sent back to base
  - The player's turn ends immediately
  - The third 6 is not used
  - If no piece was moved with the second 6, no penalty applies

#### 4. Fixed Path Examples in Documentation
- Corrected Yellow's path: 5 → ... → 68 → [enters home directly]
- Yellow does NOT wrap around after 68 (that's their home entry point)
- Added example paths for all colors

### Implementation Details

#### Modified Files

**1. [RULES.md](RULES.md)**
- Updated "Entering the Board" section: roll 5 instead of 6
- Added comprehensive "Rolling a 6" section with bonus turn rules
- Added "Three Consecutive 6s Rule" section
- Fixed example paths for all player colors
- Updated "Blocked Moves" to mention bonus turns continue even without legal moves

**2. [parchis/game.py](parchis/game.py)**
- Changed `get_legal_moves()`: check for dice_roll == 5 instead of 6 for entering pieces
- Complete rewrite of `play_turn()` method:
  - Now handles multiple rolls in a single turn
  - Tracks consecutive sixes
  - Implements three-6s penalty
  - Returns turn_info with 'rolls' array instead of single roll
  - Applies penalty by removing piece from board and sending to base
- Updated `format_turn_info()` to display multiple rolls with "Bonus Roll" labels
- Displays three-6s penalty message when applicable

**3. [parchis/logger.py](parchis/logger.py)**
- Updated `log_turn()` to handle new structure:
  - Logs array of rolls instead of single roll
  - Each roll contains: dice_roll, legal_moves_count, move
  - Adds three_sixes_penalty boolean flag
  - Logs penalty_piece information if penalty occurred

**4. [README.md](README.md)**
- Updated Key Rules Summary:
  - Changed "Roll a 6" to "Roll a 5"
  - Added "Rolling a 6 grants a bonus turn"
  - Added "Three consecutive 6s: piece from second 6 is captured"

### Game Log Format Changes

**Previous format (v1.1):**
```json
{
  "turn_number": 1,
  "player_id": 0,
  "player_color": "YELLOW",
  "dice_roll": 5,
  "legal_moves_count": 1,
  "move": { ... }
}
```

**New format (v1.2):**
```json
{
  "turn_number": 1,
  "player_id": 0,
  "player_color": "YELLOW",
  "rolls": [
    {
      "dice_roll": 6,
      "legal_moves_count": 1,
      "move": { ... }
    },
    {
      "dice_roll": 4,
      "legal_moves_count": 1,
      "move": { ... }
    }
  ],
  "three_sixes_penalty": false
}
```

### Testing Results

All tests pass successfully:
- ✓ Entry with roll of 5 works correctly
- ✓ Bonus turns trigger when rolling 6
- ✓ Multiple rolls in single turn logged correctly
- ✓ Three-6s penalty logic implemented (captures piece from second 6)
- ✓ Complete games with 2 and 4 players finish successfully
- ✓ Game logs show correct roll structure

### Gameplay Impact

These changes significantly affect gameplay:

**Piece Entry:**
- Rolling 5 to enter pieces (same 1/6 probability as rolling 6)
- Entry dice value now differs from bonus turn dice value (5 vs 6)
- More strategic differentiation between entry and movement

**Bonus Turns Create Momentum:**
- Players who roll 6s get multiple moves
- Can lead to dramatic comebacks or runaways
- Adds excitement and unpredictability

**Three-6s Rule Adds Risk:**
- Lucky streaks (three 6s) have a penalty
- Balances the advantage of bonus turns
- Adds strategic tension when rolling 6s

### Backward Compatibility

**Breaking Changes:**
- Game logs from v1.1 have different structure (single roll vs rolls array)
- Entry condition changed (roll 5 vs roll 6)
- Turn dynamics completely different due to bonus turns

Old logs can still be read but represent different game rules.

### Migration from v1.1

If you have existing code or tools that read game logs:
- Update to expect `rolls` array instead of single `dice_roll`/`move` fields
- Handle `three_sixes_penalty` field
- Account for `penalty_piece` field when penalty occurred
- Note that turns can now have multiple actions

### Performance Notes

- Games may take fewer turns overall due to bonus turns allowing more moves
- Turn processing is slightly more complex (loop for bonus turns)
- Log files will be larger (multiple rolls per turn)

### Known Limitations

The three-6s penalty only applies to the piece moved with the second 6. If that piece:
- Was finished (reached position 76), no penalty applies
- Was in base when second 6 was rolled, no penalty applies
- Had no legal move on second 6, no penalty applies

This matches the rule specification.
