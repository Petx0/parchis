# Bonus Move Deadlock Fix

## Problem Summary

Training would result in 25% timeout rate where games reached 10,000+ steps without finishing. Investigation revealed these were not true game deadlocks, but a bug in the bonus move system.

## Root Cause

When a piece finished or captured an opponent, the environment would automatically set a pending bonus move (10 or 20 squares). However, if there were **no legal moves** with that bonus (e.g., a piece at position 75 in home column with a 10-square bonus would overshoot position 76), the agent would get stuck in an infinite loop:

1. Agent finishes a piece at position 76
2. Environment sets `pending_bonus = {'type': 'finish_bonus', 'squares': 10}`
3. Agent's only remaining piece is at position 75 in home column
4. A 10-square move from 75 would land at position 85 (overshooting position 76)
5. `get_legal_moves()` returns empty list (no legal moves)
6. Agent cannot make a move, but turn doesn't end because `pending_bonus` is set
7. **Infinite loop** - agent keeps trying to make a move that doesn't exist

## Example Deadlock State (Episode 12)

```
Player 0 (GREEN) - Agent's turn:
  Piece 0: position=76, finished=True
  Piece 1: position=76, finished=True
  Piece 2: position=75, in_home_column=True  ← Only piece left
  Piece 3: position=76, finished=True

Pending bonus: {'type': 'finish_bonus', 'squares': 10}
Legal moves available: 0  ← BUG! No legal moves but bonus is pending
```

The piece at position 75 needs only 1 square to finish, but has a 10-square bonus. Since you cannot overshoot in the home column, there are no legal moves, but the environment expects the agent to make one.

## The Fix

**File**: `parchis/env.py` lines 207-221

Check if there are legal moves BEFORE setting a pending bonus:

```python
# Set pending bonus if triggered AND if there are legal moves with it
if new_bonus is not None:
    # Check if there are legal moves with this bonus
    legal_moves_with_bonus = self.game.get_legal_moves(current_player, new_bonus['squares'])
    if len(legal_moves_with_bonus) > 0:
        self.pending_bonus = new_bonus
        # Don't advance to next player, same player gets bonus move
    else:
        # Bonus triggered but no legal moves - skip bonus
        self.pending_bonus = None
        self.bonus_chain_count = 0
        # Turn is over, continue to next player
else:
    # No bonus, reset chain count
    self.bonus_chain_count = 0
```

## Impact

### Before Fix (with bug):
- **Wins**: 6/20 (30.0%)
- **Losses**: 9/20 (45.0%)
- **Timeouts**: 5/20 (25.0%) ❌
- **Mean episode length**: 2,573 steps
- Games would run for 10,000 steps without finishing

### After Fix:
- **Wins**: 13/20 (65.0%) ✅ (+117% improvement)
- **Losses**: 7/20 (35.0%)
- **Timeouts**: 0/20 (0.0%) ✅ (completely eliminated)
- **Mean episode length**: 95 steps ✅ (27x faster)
- All games complete normally

## Why This Matters

1. **Eliminates false deadlocks** - Games that appeared stuck were actually bugs, not game logic issues
2. **Dramatically improves training efficiency** - Games complete in ~95 steps instead of timing out at 10,000
3. **Better agent performance** - Win rate doubled from 30% to 65%
4. **Correct game rules** - The environment now properly handles the rule that you cannot overshoot in the home column

## Technical Details

### Home Column Rules
- Home column positions: 69-76
- Position 76 is the final position
- You **must land exactly** on position 76 to finish
- You **cannot overshoot** position 76

### When Bonuses Are Skipped
A bonus move is skipped (not offered to agent) when:
1. **Finish bonus (10 squares)**: All remaining pieces would overshoot if moved
2. **Capture bonus (20 squares)**: All pieces are either:
   - Finished
   - In base (can only enter with a dice roll of 5)
   - Would land on a position blocked by a blockade
   - Would overshoot if in home column

### Game State When Bonus Is Skipped
When a bonus has no legal moves:
- `pending_bonus` is set to `None`
- `bonus_chain_count` is reset to 0
- `turn_over` becomes `True` (since `pending_bonus is None`)
- Game advances to next player normally

## Testing

To verify the fix:
```bash
python -m parchis.train_quick
```

Expected results:
- Training completes in ~5 seconds
- Evaluation shows 0 timeouts
- Win rate should be >50% even with minimal training
- Mean episode length should be <150 steps

## Related Files

- **parchis/env.py**: Main fix (lines 207-221)
- **parchis/game.py**: `get_legal_moves()` correctly handles overshooting (lines 245-258)
- **parchis/train_ppo.py**: Debugging code added for timeout detection (lines 253-276)

## Notes

- This was the root cause of the "25% timeout rate" issue
- The timeouts were not caused by game deadlocks (blockades, etc.)
- The timeouts were not caused by agent behavior
- The timeouts were caused by a bug in the environment's bonus handling logic
- With this fix, the environment now correctly implements Parchís rules for bonus moves
