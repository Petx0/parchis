# Changelog - Version 1.3

## Bonus Move System Implementation (2026-01-11)

This update implements a comprehensive bonus move system that adds significant strategic depth to the game.

### New Features

#### 1. Capture Bonus (20 squares)
- **Trigger**: When a player captures an opponent's piece
- **Effect**: Immediately receive a bonus move of 20 squares
- **Flexibility**: Can choose ANY piece on the board (not just the capturing piece)
- **Mandatory**: Must use the bonus if any legal 20-square move exists
- **Chaining**: Can trigger additional bonuses (capture → finish, capture → capture)

#### 2. Finish Bonus (10 squares)
- **Trigger**: When a piece reaches the final square (position 76)
- **Effect**: Immediately receive a bonus move of 10 squares
- **Flexibility**: Can choose any remaining piece on the board
- **Mandatory**: Must use the bonus if any legal 10-square move exists
- **Chaining**: Can trigger additional bonuses (finish → capture, finish → finish)

#### 3. Bonus Chaining
- Bonuses can chain indefinitely within a single turn
- Chain examples:
  - Capture → 20-square move → Capture → 20-square move → Finish → 10-square move
  - Finish → 10-square move → Capture → 20-square move → Finish → 10-square move
- All bonuses resolved immediately before next action
- All bonuses occur within the same turn

#### 4. Interaction with Rolling 6
- **Execution Order**: Bonuses execute BEFORE the next roll from rolling a 6
- Example turn sequence:
  1. Roll 6
  2. Move piece (capture occurs)
  3. Execute 20-square bonus move
  4. Bonus chains resolve (if any)
  5. Roll again (because of the 6)
- Three-6s rule still applies after bonuses are processed

### Implementation Details

#### Modified Files

**1. [RULES.md](RULES.md)** (lines 169-192)
Added comprehensive "Bonus Moves" section documenting:
- Capture bonus (20 squares) rules
- Finish bonus (10 squares) rules
- Bonus chaining mechanics
- Mandatory bonus usage
- All normal movement rules apply to bonuses

**2. [parchis/game.py](parchis/game.py)**

Added new methods:
- `handle_bonus_moves(player, initial_move_info, turn_info)` (lines 355-392)
  - Main bonus processing loop
  - Handles bonus chaining automatically
  - Checks for both capture and finish bonuses

- `_execute_bonus_move(player, bonus_squares, bonus_type, turn_info)` (lines 394-450)
  - Executes a single bonus move
  - Gets legal moves for bonus squares
  - Records bonus attempt even if no legal moves
  - Returns move_info for chaining

Updated `play_turn()` method (line 510):
- Calls `handle_bonus_moves()` immediately after each move
- Bonuses execute before checking for rolling-6 bonus turn
- Win condition checked after all bonuses complete

Updated `format_turn_info()` (lines 609-649):
- Displays bonus moves with clear labels
- Shows "[CAPTURE BONUS - 20 squares]" or "[FINISH BONUS - 10 squares]"
- Distinguishes between dice rolls and bonus moves

**3. [parchis/logger.py](parchis/logger.py)** (lines 58-92)

Updated `log_turn()` method:
- Detects bonus moves by checking for 'bonus_type' field
- Logs bonus moves with `bonus_type` and `bonus_squares` fields
- Maintains same structure for regular dice rolls
- Compatible with existing log analysis tools

**4. [test_game.py](test_game.py)** (lines 548-800)

Added comprehensive `test_bonus_moves()` function with 6 test scenarios:
1. Capture bonus (20 squares) awarded
2. Finish bonus (10 squares) awarded
3. Bonus chaining logic implemented
4. Bonus not executed when no legal moves available
5. Bonus execution order with rolling 6 (bonuses before next roll)
6. Three consecutive 6s rule with bonuses (applies after bonuses)

### Game Log Format Changes

**Previous format (v1.2):**
```json
{
  "rolls": [
    {
      "dice_roll": 5,
      "legal_moves_count": 1,
      "move": { ... }
    }
  ]
}
```

**New format (v1.3) with bonuses:**
```json
{
  "rolls": [
    {
      "dice_roll": 5,
      "legal_moves_count": 1,
      "move": { ... }
    },
    {
      "bonus_type": "capture_bonus",
      "bonus_squares": 20,
      "legal_moves_count": 3,
      "move": { ... }
    },
    {
      "bonus_type": "finish_bonus",
      "bonus_squares": 10,
      "legal_moves_count": 2,
      "move": { ... }
    }
  ]
}
```

### Testing Results

All tests pass successfully:
- ✓ Capture bonus (20 squares) awarded
- ✓ Finish bonus (10 squares) awarded
- ✓ Bonus chaining logic implemented
- ✓ Bonus not executed when no legal moves available
- ✓ Bonus executes before next roll from rolling 6
- ✓ Three-6s rule logic verified (applies after bonuses)
- ✓ Complete game (201 turns, YELLOW wins)
- ✓ Bonuses logged correctly in game logs

### Gameplay Impact

These changes significantly affect gameplay strategy:

**Capture Incentive:**
- Capturing now provides massive 20-square bonus
- Creates aggressive play strategies
- Risk/reward calculation for capture attempts
- Can lead to dramatic position swings

**Finish Strategy:**
- Finishing a piece provides 10-square bonus
- Encourages racing pieces to finish
- Can help advance other pieces significantly
- Strategic timing of finish moves matters

**Bonus Chaining:**
- Spectacular multi-bonus chains possible
- A single capture can cascade into multiple moves
- Adds excitement and unpredictability
- Can completely change game state in one turn

**Interaction with Blockades:**
- Bonuses must respect blockades (cannot cross)
- Strategic blockade placement more important
- Bonuses can help escape or create blockades

**Synergy with Rolling 6:**
- Rolling 6 + capture = bonus move + another roll
- Can lead to very long turns with multiple actions
- Three-6s penalty still provides balance

### Backward Compatibility

**Breaking Changes:**
- Game logs from v1.2 won't show bonus moves
- Turn structure different (can have many more actions per turn)
- Game length may be shorter due to bonuses accelerating gameplay

Old logs can still be read but represent different gameplay dynamics.

### Performance Notes

- Bonus processing is recursive but terminates when no more bonuses
- Turn processing slightly more complex (bonus loop)
- Game length typically shorter (bonuses accelerate movement)
- Log files may be larger (more actions per turn)

### Known Behaviors

**Bonus Processing:**
- If no legal bonus move exists, bonus is logged but not executed
- Bonuses are mandatory if legal moves exist
- Player chooses which piece to move for bonus (random for AI players)

**Edge Cases:**
- Bonus from last piece finishing has no effect (no pieces left to move)
- Bonus blocked by own blockades (cannot cross)
- Multiple chained bonuses all occur in same turn
- Rolling 6 bonus turn happens AFTER all bonuses resolve

### Future Enhancements

- Statistics tracking for bonus chains
- Bonus chain visualization in logs
- Strategic AI that optimizes for bonus opportunities
- Bonus move highlights in future GUI
