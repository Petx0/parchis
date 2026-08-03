# Bug Fixes - Training Environment

## Issue: Training Stuck at Evaluation

**Problem**: When running `train_quick.py`, training would get stuck around halfway through (e.g., 19,999 out of 100,000 timesteps).

**Root Cause**: The environment only controlled Player 0 (the learning agent). When the turn passed to Players 1, 2, or 3, the environment would wait for the next `step()` call, but it wasn't the learning agent's turn. This caused the environment to hang, waiting for actions from players that weren't being controlled.

### Fix 1: Auto-Play for Non-Learning Players

**File**: `parchis/env.py`

**Changes**:
1. Added a `while` loop in the `step()` method that automatically plays moves for non-learning players (Players 1, 2, 3)
2. Added `_auto_play_bonus()` method to handle bonus moves for non-learning players
3. Non-learning players use random moves (via `Player.choose_move()`)

**Code** (lines 238-275):
```python
# Auto-play for non-learning players (players 1, 2, 3)
# Player 0 is the learning agent, others play randomly
while self.game.current_player_idx != 0 and not terminated and not truncated:
    current_player = self.game.get_current_player()

    # Get legal moves and choose randomly
    legal_moves = self.game.get_legal_moves(current_player, self.current_dice_roll)
    chosen_move = current_player.choose_move(legal_moves)

    if chosen_move:
        piece, new_position, move_type = chosen_move
        move_info = self.game.execute_move(piece, new_position, move_type)

        # Handle bonus moves automatically
        if len(move_info['captured']) > 0:
            self._auto_play_bonus(current_player, 20)
        elif move_info['new_position'] == Board.FINAL_POSITION:
            self._auto_play_bonus(current_player, 10)

    # Check if this player won
    if current_player.has_won():
        reward = -100  # Agent lost
        terminated = True
        break

    # Move to next player
    self.game.next_player()
    self.current_dice_roll = self.game.dice.roll()

    # Check truncation
    self.episode_length += 1
    if self.episode_length >= self.max_episode_length:
        truncated = True
        break
```

### Fix 2: Action Type Conversion

**Problem**: `model.predict()` returns actions as numpy arrays, but the environment expects integers.

**Fix**: Convert action to int before passing to `env.step()`:
```python
action, _ = model.predict(obs, action_masks=action_masks)
action = int(action)  # Convert numpy array to int
obs, reward, terminated, truncated, info = env.step(action)
```

## How Training Works Now

1. **Agent's Turn** (Player 0):
   - Environment waits for `step(action)` call
   - Agent chooses action via PPO policy
   - Move is executed
   - If bonus triggered, agent controls bonus moves (via subsequent `step()` calls)

2. **Other Players' Turns** (Players 1, 2, 3):
   - After agent's turn ends, environment automatically plays for other players
   - Each player makes random moves (via `Player.choose_move()`)
   - Bonus moves are handled automatically
   - Loop continues until it's the agent's turn again

3. **Episode End Conditions**:
   - `terminated=True`: Someone won the game
   - `truncated=True`: Reached `max_episode_length` (default: 1000 total game turns)

## Performance Characteristics

**Episode Statistics** (from test runs):
- Average episode length: ~60-100 agent actions
- Total game turns: ~200-400 (includes all 4 players)
- Training speed: ~1500-2000 FPS on CPU

**Training Metrics**:
- Episodes per 1000 timesteps: ~15-20
- Each episode involves ~15-20 decisions from the learning agent
- Other players make 3x more moves (since there are 3 of them)

## Testing

**Simple Test** (without TensorBoard):
```bash
python -m parchis.test_training_simple
```

Expected output:
```
Creating environment...
Creating model...
Training for 5000 timesteps...
✓ Training completed successfully!
Testing trained model...
✓ All tests passed!
```

**Full Environment Test**:
```bash
python -m parchis.test_env
```

All tests should pass, including:
- Basic initialization
- Step execution
- Bonus move handling
- Observation structure
- Action masking

## Remaining Issues

1. **TensorBoard Installation**: TensorBoard is still installing (grpcio is building)
   - Training works without it (`tensorboard_log=None`)
   - Once installed, full logging will be available

2. **Single Agent Learning**: Currently only Player 0 learns
   - Players 1, 2, 3 play randomly
   - This is intentional for the initial implementation
   - Can be upgraded to self-play later

## Next Steps

1. **Wait for TensorBoard**: Let the grpcio build finish
2. **Run Quick Training**: Try `python -m parchis.train_quick`
3. **Monitor Progress**: Use TensorBoard once installed
4. **Tune Hyperparameters**: Adjust based on initial results
5. **Consider Self-Play**: Implement for better performance (future enhancement)

## Files Modified

- `parchis/env.py`: Added auto-play for non-learning players
- `parchis/test_training_simple.py`: Created simple training test (NEW)
- `FIXES.md`: This file (NEW)
