import pygame
import numpy as np
import random
import math
from collections import defaultdict

# --- New: Difficulty Settings ---
DIFFICULTY_SETTINGS = {
    'easy': {'iterations': 50},
    'medium': {'iterations': 200}, # Original value
    'hard': {'iterations': 1000},
    'very_hard': {'iterations': 5000} # For comparison, will be slow
}

CURRENT_DIFFICULTY = 'medium' # Set initial difficulty
# --- End of Difficulty Settings ---

# Constants
BOARD_SIZE = 9
GRID_SIZE = 50
MARGIN = 40
WINDOW_SIZE = GRID_SIZE * (BOARD_SIZE - 1) + 2 * MARGIN
STONE_RADIUS = GRID_SIZE // 2 - 2
BACKGROUND = (220, 180, 140)
LINE_COLOR = (0, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)

# Winning threshold: First player to capture this many stones wins
WINNING_CAPTURES_THRESHOLD = 15  # Adjust this value as needed

class GoState:
    def __init__(self, board=None, next_player=0, captures=None):
        self.board = board if board is not None else np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=int)
        self.next_player = next_player  # 0: Black (human), 1: White (AI)
        self.captured_stones = captures if captures is not None else {0: 0, 1: 0}
        
    def get_legal_moves(self):
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == -1:
                    moves.append((r, c))
        return moves
    
    def play_move(self, r, c):
        new_board = np.copy(self.board)
        new_board[r][c] = self.next_player
        
        # Remove captured opponent groups
        opponent = 1 - self.next_player
        captured = 0
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and new_board[nr][nc] == opponent:
                group, libs = self._get_group(nr, nc, new_board)
                if len(libs) == 0:  # Captured group
                    for gr, gc in group:
                        new_board[gr][gc] = -1
                    captured += len(group)
        
        # Check if move itself creates liberties
        group, libs = self._get_group(r, c, new_board)
        if len(libs) == 0:  # Suicide move - revert
            return None
            
        new_captures = dict(self.captured_stones)
        new_captures[self.next_player] += captured
        new_state = GoState(new_board, 1 - self.next_player, new_captures)
        return new_state
    
    def _get_group(self, r, c, board):
        color = board[r][c]
        if color == -1:
            return set(), set()
        
        group = set()
        frontier = [(r, c)]
        libs = set()
        
        while frontier:
            cr, cc = frontier.pop()
            if (cr, cc) in group:
                continue
            group.add((cr, cc))
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    neighbor = board[nr][nc]
                    if neighbor == -1:
                        libs.add((nr, nc))
                    elif neighbor == color and (nr, nc) not in group:
                        frontier.append((nr, nc))
        return group, libs
    
    def is_game_over(self):
        # Simple check: if board is full or someone has reached capture threshold
        if len(self.get_legal_moves()) == 0:
            return True
        # Check if either player has captured enough stones to win
        if self.captured_stones[0] >= WINNING_CAPTURES_THRESHOLD or \
           self.captured_stones[1] >= WINNING_CAPTURES_THRESHOLD:
            return True
        return False

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.wins = 0
        self.visits = 0
        self.untried_actions = state.get_legal_moves()
    
    def ucb1(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

def mcts(root_state, iterations=1000): # Accept iterations as argument
    root = MCTSNode(root_state)
    
    for _ in range(iterations):
        node = root
        
        # Selection
        while node.untried_actions == [] and node.children:
            node = max(node.children.values(), key=lambda n: n.ucb1())
        
        # Expansion
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            new_state = node.state.play_move(*action)
            if new_state:
                node.untried_actions.remove(action)
                child = MCTSNode(new_state, parent=node)
                node.children[action] = child
                node = child
        
        # Simulation
        rollout_state = node.state
        while rollout_state and not rollout_state.is_game_over():
            legal_moves = rollout_state.get_legal_moves()
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            rollout_state = rollout_state.play_move(*move)
            if rollout_state is None:  # Invalid move
                continue
        
        # Backpropagation
        while node:
            node.visits += 1
            if rollout_state and rollout_state.next_player != node.state.next_player:  # Current player won
                node.wins += 1
            node = node.parent
    
    if root.children:
        return max(root.children.items(), key=lambda item: item[1].visits)[0]
    else:
        # Fallback to random legal move if no children were created
        legal_moves = root_state.get_legal_moves()
        return random.choice(legal_moves) if legal_moves else None

class GoGame:
    def __init__(self):
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=int)
        self.current_player = 0  # 0: Human (Black), 1: AI (White)
        self.game_history = []
        self.ai_thinking = False
        self.game_over = False
        self.winner = None # None, 0 (Black/Human), 1 (White/AI)
        self.captured_stones = {0: 0, 1: 0} # Track captures
        self.difficulty = CURRENT_DIFFICULTY # Store current difficulty
        
    def place_stone(self, x, y):
        if self.board[x][y] != -1 or self.ai_thinking or self.game_over:
            return False
            
        # Update board and captures
        self.board[x][y] = self.current_player
        
        # Check for captures after placing stone
        opponent = 1 - self.current_player
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = x + dr, y + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr][nc] == opponent:
                group, libs = self._get_group(nr, nc, self.board)
                if len(libs) == 0:  # Captured group
                    for gr, gc in group:
                        self.board[gr][gc] = -1
                    self.captured_stones[self.current_player] += len(group)
        
        self.game_history.append((x, y, self.current_player))
        
        # Check for win condition
        if self.captured_stones[self.current_player] >= WINNING_CAPTURES_THRESHOLD:
            self.game_over = True
            self.winner = self.current_player
            return True

        # Switch to AI player if game continues
        self.current_player = 1
        self.ai_thinking = True
        return True

    def _get_group(self, r, c, board):
        """Helper function to get connected group and liberties"""
        color = board[r][c]
        if color == -1:
            return set(), set()
        
        group = set()
        frontier = [(r, c)]
        libs = set()
        
        while frontier:
            cr, cc = frontier.pop()
            if (cr, cc) in group:
                continue
            group.add((cr, cc))
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    neighbor = board[nr][nc]
                    if neighbor == -1:
                        libs.add((nr, nc))
                    elif neighbor == color and (nr, nc) not in group:
                        frontier.append((nr, nc))
        return group, libs

    def ai_move(self):
        if self.ai_thinking and not self.game_over:
            # Create state for AI considering captures
            state = GoState(self.board, 1, self.captured_stones)
            # Get iterations from settings based on current difficulty
            iterations = DIFFICULTY_SETTINGS[self.difficulty]['iterations']
            best_move = mcts(state, iterations=iterations) # Pass the dynamic iterations value
            
            if best_move:
                r, c = best_move
                self.board[r][c] = 1
                
                # Check for captures after AI move
                opponent = 1 - 1 # Human player
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr][nc] == opponent:
                        group, libs = self._get_group(nr, nc, self.board)
                        if len(libs) == 0:  # Captured group
                            for gr, gc in group:
                                self.board[gr][gc] = -1
                            self.captured_stones[1] += len(group)
                
                self.game_history.append((r, c, 1))
                
                # Check for win condition
                if self.captured_stones[1] >= WINNING_CAPTURES_THRESHOLD:
                    self.game_over = True
                    self.winner = 1
                else:
                    self.current_player = 0
                    
            self.ai_thinking = False

    def change_difficulty(self, new_difficulty):
        """Change the AI difficulty level."""
        if new_difficulty in DIFFICULTY_SETTINGS:
            self.difficulty = new_difficulty
            print(f"Difficulty changed to {new_difficulty.capitalize()}")

def draw_board(screen, game):
    screen.fill(BACKGROUND)
    
    # Draw grid lines
    for i in range(BOARD_SIZE):
        pos = MARGIN + i * GRID_SIZE
        pygame.draw.line(screen, LINE_COLOR, (MARGIN, pos), 
                         (WINDOW_SIZE - MARGIN, pos), 2)
        pygame.draw.line(screen, LINE_COLOR, (pos, MARGIN), 
                         (pos, WINDOW_SIZE - MARGIN), 2)
    
    # Draw stones
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if game.board[x][y] != -1:
                color = BLACK if game.board[x][y] == 0 else WHITE
                center = (MARGIN + x * GRID_SIZE, MARGIN + y * GRID_SIZE)
                pygame.draw.circle(screen, color, center, STONE_RADIUS)
                if color == WHITE:
                    pygame.draw.circle(screen, BLACK, center, STONE_RADIUS, 1)
    
    # Draw captures info
    font = pygame.font.SysFont(None, 24)
    black_text = font.render(f"Black Captures: {game.captured_stones[0]}", True, TEXT_COLOR)
    white_text = font.render(f"White Captures: {game.captured_stones[1]}", True, TEXT_COLOR)
    difficulty_text = font.render(f"AI Difficulty: {game.difficulty.capitalize()}", True, TEXT_COLOR)
    screen.blit(black_text, (10, 10))
    screen.blit(white_text, (WINDOW_SIZE - white_text.get_width() - 10, 10))
    screen.blit(difficulty_text, (WINDOW_SIZE//2 - difficulty_text.get_width()//2, 10)) # Centered at top
    
    # Draw thinking indicator
    if game.ai_thinking:
        font = pygame.font.SysFont(None, 36)
        text = font.render("AI is thinking...", True, TEXT_COLOR)
        screen.blit(text, (WINDOW_SIZE//2 - text.get_width()//2, 40)) # Moved down slightly
    
    # Draw game over message
    if game.game_over:
        font = pygame.font.SysFont(None, 48)
        if game.winner is not None:
            winner_text = f"{'Human (Black)' if game.winner == 0 else 'AI (White)'} Wins!"
        else:
            winner_text = "Game Over - Draw?"
        text = font.render(winner_text, True, TEXT_COLOR)
        screen.blit(text, (WINDOW_SIZE//2 - text.get_width()//2, WINDOW_SIZE//2 - text.get_height()//2))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Capture Go - Human vs AI (with Difficulty Levels)")
    clock = pygame.time.Clock()
    game = GoGame()
    
    ai_timer = 0
    ai_delay = 30  # frames to wait before AI move
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and game.current_player == 0 and not game.ai_thinking and not game.game_over:
                x, y = event.pos
                grid_x = round((x - MARGIN) / GRID_SIZE)
                grid_y = round((y - MARGIN) / GRID_SIZE)
                
                if 0 <= grid_x < BOARD_SIZE and 0 <= grid_y < BOARD_SIZE:
                    game.place_stone(grid_x, grid_y)
                    ai_timer = 0  # Reset timer after human move
            # --- New: Handle Key Presses for Difficulty ---
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    game.change_difficulty('easy')
                elif event.key == pygame.K_2:
                    game.change_difficulty('medium')
                elif event.key == pygame.K_3:
                    game.change_difficulty('hard')
                elif event.key == pygame.K_4:
                    game.change_difficulty('very_hard')
            # --- End of Key Press Handling ---
        
        # Handle AI move after delay
        if game.ai_thinking and not game.game_over:
            ai_timer += 1
            if ai_timer >= ai_delay:
                game.ai_move()
        
        draw_board(screen, game)
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()