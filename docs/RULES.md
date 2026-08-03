# Parchís Game Rules

## Game Overview
Parchís (also known as Ludo) is a race game for 2-4 players where each player aims to move all their pieces from their starting position around the board and into their home column to win.

**Player Colors:**
- **4 players**: Yellow, Blue, Red, Green (all colors)
- **3 players**: Yellow, Blue, Red (first three colors)
- **2 players**: Red vs Yellow, or Blue vs Green (always opposite colors for balanced gameplay)

## Board Design

### Board Layout
The board consists of:
- **4 colored sections**: Yellow, Blue, Red, Green (one per player)
- **68 main track squares**: A circular path around the board (numbered 1-68)
- **4 home columns**: Each with 8 squares leading to the final position (numbered 69-76 per player)
- **4 starting squares**: One per player color at specific positions on the main track
- **Safe squares**: Certain positions where pieces cannot be captured

### Square Numbering System
The board uses absolute numbering for the circular track (1-68) and player-specific home columns:

**Main Circular Track (1-68):**
- Yellow starts at square 5
- Blue starts at square 22
- Red starts at square 39
- Green starts at square 56

**Home Column Entry Points:**
- Yellow enters home column after square 68 (completes full lap from 5 → 68)
- Blue enters home column after square 17 (completes full lap from 22 → 17)
- Red enters home column after square 34 (completes full lap from 39 → 34)
- Green enters home column after square 51 (completes full lap from 56 → 51)

**Home Columns (8 squares each):**
- Squares 69-76 for each player's home column
- Square 76 is the final victory position

**Example Path for Yellow:**
5 → 6 → 7 → ... → 68 → [enters home] → 69 → 70 → 71 → 72 → 73 → 74 → 75 → 76 (finish)

**Example Path for Blue:**
22 → 23 → 24 → ... → 68 → [wraps to 1] → 2 → 3 → ... → 17 → [enters home] → 69 → 70 → 71 → 72 → 73 → 74 → 75 → 76 (finish)

**Example Path for Red:**
39 → 40 → 41 → ... → 68 → [wraps to 1] → 2 → 3 → ... → 34 → [enters home] → 69 → 70 → 71 → 72 → 73 → 74 → 75 → 76 (finish)

**Example Path for Green:**
56 → 57 → 58 → ... → 68 → [wraps to 1] → 2 → 3 → ... → 51 → [enters home] → 69 → 70 → 71 → 72 → 73 → 74 → 75 → 76 (finish)

### Safe Squares
The following squares on the main track are safe (pieces cannot be captured here):
- **Starting squares**: 5 (Yellow), 22 (Blue), 39 (Red), 56 (Green)
- **Additional safe squares**: 12, 17, 29, 34, 46, 51, 63, 68

## Game Rules - Version 1

### Setup
1. **Players**: 2-4 players, each assigned a color (Yellow, Blue, Red, Green)
2. **Starting position**: Each player starts with **1 piece already on their starting square**
   - Yellow: square 5
   - Blue: square 22
   - Red: square 39
   - Green: square 56
3. **Remaining pieces**: The other 3 pieces start in the "base" (off the board)
4. **Turn order**: Players take turns in clockwise order (Yellow → Blue → Red → Green)

### Dice
- **Single 6-sided die**: Players roll one die per turn
- **Values**: 1-6

### Turn Structure
1. Player rolls the die
2. Player must make a legal move if possible
3. Turn passes to the next player

### Movement Rules

#### Entering the Board
When a player rolls a **5**, they **must** bring a new piece from base to their starting square if they have pieces in base and the move is legal. This takes precedence over moving pieces already on the board.

The rules for entry depend on what's currently at the starting square:

1. **Starting square is empty**: New piece moves to starting square
2. **One own piece at starting square**: New piece can enter; both pieces occupy the safe starting square
3. **One opponent piece at starting square**: New piece can enter; both pieces occupy the safe starting square (no capture)
4. **Two own pieces at starting square**: Cannot bring new piece out; must use the 5 to move another piece
5. **Two pieces: one own, one opponent**: Can bring new piece out and capture the opponent's piece
6. **Two opponent pieces**: Can bring new piece out and capture the opponent piece that was **most recently** moved to that square

Key points:
- Starting squares are safe and can hold multiple pieces regardless of color
- When entering causes a capture, only happens with two pieces present (rules 5-6)
- **Mandatory entry**: If rolling 5 with pieces in base, must enter if legal (cannot choose to move other pieces instead)
- If entry is illegal (e.g., two own pieces blocking), then can move other pieces

#### Moving Pieces
- A piece moves forward by the number shown on the die
- Pieces move along the main circular track (1-68), wrapping around (after 68 comes 1)
- When a piece reaches its home entry point, it enters the home column (69-76)
- **Piece stacking rules**:
  - Two pieces of the same color can occupy the same square (creating a stack)
  - A square can hold at most two pieces total
  - Moving to a square that already has two pieces (regardless of color or safe status) is illegal
  - When a piece lands on a square with one opponent piece, the opponent is captured (unless it's a safe square)
  - When a piece lands on a square with one own piece, both pieces remain (stack)

#### Capturing
- If a piece lands on a square occupied by an opponent's piece (and it's not a safe square):
  - The opponent's piece is **captured** and returns to their base
  - The captured piece must re-enter the board with a roll of 5
- Pieces on **safe squares** cannot be captured
- Players **cannot capture their own pieces**
- **Home columns are capture-free**: Pieces in home columns (69-76) cannot be captured
  - Each player's home column is independent and private to that player
  - Home column squares can hold a maximum of two pieces of the same color

#### Entering Home Column
- Each player enters their home column after completing one full lap around the board:
  - Yellow: enters home after square 68
  - Blue: enters home after square 17
  - Red: enters home after square 34
  - Green: enters home after square 51
- Home columns have 8 squares (69-76)
- Pieces must move through the home column to reach the final square (76)
- **Exact roll required**: To reach square 76, the piece must land exactly on it
  - If the roll would move the piece beyond square 76, the move is illegal and cannot be made

#### Winning
- A player wins when **all 4 pieces** reach the final square (square 76)
- The game ends immediately when a player wins

### Special Rules

#### Rolling a 5
- **Mandatory entry**: If a player has pieces in base and rolls a 5, they **must** bring a piece from base to the starting square if the move is legal
- Entry takes precedence over moving pieces already on the board
- If entry is not possible (e.g., two own pieces blocking the starting square), then the player can move an existing piece forward 5 squares
- When entering with two pieces already at starting square, may capture opponent pieces (see "Entering the Board" for details)

#### Rolling a 6
- **Bonus Turn**: When a player rolls a 6, they get to roll again after completing their move
- This bonus turn applies whether or not the player had a legal move
- **Standard move**: The player can move any piece forward 6 squares (following normal movement rules)
- **Special rule - All pieces out of base**: When a player has 0 pieces remaining in base and rolls a 6, they move 7 squares instead of 6
  - This applies only when all 4 pieces are either on the board or finished
  - The bonus turn still applies (player rolls again after moving 7 squares)

#### Three Consecutive 6s Rule
- If a player rolls three 6s in a row on the same turn:
  - The piece that was moved with the second 6 is **captured** and sent back to base
  - The player's turn ends immediately (they do not get to use the third 6)
  - If no piece was moved with the second 6 (no legal moves), no piece is captured

**Exceptions:**
1. **No bonus triggered**: When a piece is sent back to base due to the three-6s penalty, it does NOT trigger a capture bonus (no 20-square bonus move)
2. **Home column entry protection**: If the piece moved with the second 6 entered the home column (moved from main track to home column), it is NOT sent back to base when the third 6 is rolled
   - The piece remains in the home column
   - The third 6 cannot be used
   - The turn ends

#### Blockades
A **blockade** is formed when two pieces of the same color occupy the same safe square on the main track.

**Blockade Rules:**
- Blockades can only form on safe squares (5, 12, 17, 22, 29, 34, 39, 46, 51, 56, 63, 68)
- When a blockade exists, **no piece** can move across it (including pieces of the same color that created the blockade)
- Movement is blocked if the path crosses through the blockade square, even with wrapping
  - Example: Yellow blockade at square 12. Yellow piece at square 10 cannot move with a roll of 4
  - Example: Blockade at square 68. Piece at square 67 cannot move with a roll of 3 (would cross 68 to reach square 1-2)
- Blockades in home columns are not possible (home columns don't use the blockade rule)

**Opening a Blockade:**
- If a player has created a blockade with their own pieces and rolls a **6**, they are **forced** to open the blockade
- The player must move one of the two pieces forming the blockade (as long as it's a legal move)
- If both pieces in the blockade have legal moves, the player can choose which one to move
- If neither piece in the blockade can make a legal move, normal movement rules apply
- This forced opening only applies when rolling a 6

#### Bonus Moves

**Capture Bonus (20 squares):**
- When a player captures an opponent's piece, they **immediately** get a bonus move of 20 squares
- The player can choose **any** of their pieces on the board to move 20 squares (not necessarily the piece that made the capture)
- All normal movement rules apply: cannot cross blockades, must land exactly on square 76, etc.
- **Mandatory**: If any piece can legally move 20 squares, the player **must** use the bonus
- If no legal 20-square move is available with any piece, the bonus goes unused
- **Chaining**: If the 20-square bonus move results in another capture, another 20-square bonus is awarded immediately
- **Chaining**: If the 20-square bonus move reaches the final square (76), a 10-square bonus is awarded immediately

**Finish Bonus (10 squares):**
- When a player moves a piece to the final square (76), they **immediately** get a bonus move of 10 squares
- The player can choose **any** of their remaining pieces on the board to move 10 squares
- All normal movement rules apply
- **Mandatory**: If any piece can legally move 10 squares, the player **must** use the bonus
- If no legal 10-square move is available, the bonus goes unused
- **Chaining**: If the 10-square bonus move results in a capture, a 20-square bonus is awarded immediately
- **Chaining**: If the 10-square bonus move reaches the final square (76), another 10-square bonus is awarded immediately

**Bonus Chaining:**
- Bonuses can chain indefinitely: Capture → 20-square move → Capture → 20-square move → Finish → 10-square move → etc.
- Each bonus must be resolved immediately before continuing
- All bonuses occur within the same turn

#### Blocked Moves
If no legal move is available (e.g., all pieces would move beyond square 76, or starting square is blocked by own piece when trying to enter), the turn is passed to the next player. However, if a 6 was rolled, the player still gets their bonus turn even if they had no legal moves.

#### Piece Priority (for Random Players)
In version 1, with random players, if multiple moves are possible, a random legal move is selected.

## Implementation Notes - Version 1

### Features
- **Random players**: All players make random legal moves
- **Game modes**:
  - Full game mode: Plays automatically from start to finish
  - Play-by-play mode: Advances one turn at a time with user input
- **Logging**: Every game is logged with complete move history
- **No visualization**: Text-based representation only (visualization planned for future versions)

### Future Enhancements
- Graphical board visualization
- Intelligent AI players (rule-based, ML-based)
- Additional rule variants (e.g., bonus turns for rolling 6, barriers with two pieces)
- Multiplayer networked play
- Statistics and game analysis tools
