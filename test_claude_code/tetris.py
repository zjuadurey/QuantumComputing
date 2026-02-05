#!/usr/bin/env python3
"""Terminal Tetris game using curses."""

import curses
import random
import time

# Tetromino shapes (each rotation state)
SHAPES = {
    'I': [[(0,0), (0,1), (0,2), (0,3)],
          [(0,0), (1,0), (2,0), (3,0)]],
    'O': [[(0,0), (0,1), (1,0), (1,1)]],
    'T': [[(0,0), (0,1), (0,2), (1,1)],
          [(0,0), (1,0), (2,0), (1,1)],
          [(0,1), (1,0), (1,1), (1,2)],
          [(0,1), (1,0), (1,1), (2,1)]],
    'S': [[(0,1), (0,2), (1,0), (1,1)],
          [(0,0), (1,0), (1,1), (2,1)]],
    'Z': [[(0,0), (0,1), (1,1), (1,2)],
          [(0,1), (1,0), (1,1), (2,0)]],
    'J': [[(0,0), (1,0), (1,1), (1,2)],
          [(0,0), (0,1), (1,0), (2,0)],
          [(0,0), (0,1), (0,2), (1,2)],
          [(0,1), (1,1), (2,0), (2,1)]],
    'L': [[(0,2), (1,0), (1,1), (1,2)],
          [(0,0), (1,0), (2,0), (2,1)],
          [(0,0), (0,1), (0,2), (1,0)],
          [(0,0), (0,1), (1,1), (2,1)]]
}

SHAPE_NAMES = list(SHAPES.keys())
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

class Tetris:
    def __init__(self):
        self.board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.lines = 0
        self.level = 1
        self.game_over = False
        self.current_piece = None
        self.current_shape = None
        self.current_rotation = 0
        self.current_x = 0
        self.current_y = 0
        self.next_shape = random.choice(SHAPE_NAMES)
        self.spawn_piece()

    def spawn_piece(self):
        self.current_shape = self.next_shape
        self.next_shape = random.choice(SHAPE_NAMES)
        self.current_rotation = 0
        self.current_piece = SHAPES[self.current_shape][0]
        self.current_x = BOARD_WIDTH // 2 - 2
        self.current_y = 0

        if not self.valid_position(self.current_x, self.current_y, self.current_piece):
            self.game_over = True

    def valid_position(self, x, y, piece):
        for dy, dx in piece:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= BOARD_WIDTH or ny >= BOARD_HEIGHT:
                return False
            if ny >= 0 and self.board[ny][nx]:
                return False
        return True

    def lock_piece(self):
        for dy, dx in self.current_piece:
            nx, ny = self.current_x + dx, self.current_y + dy
            if 0 <= ny < BOARD_HEIGHT and 0 <= nx < BOARD_WIDTH:
                self.board[ny][nx] = SHAPE_NAMES.index(self.current_shape) + 1
        self.clear_lines()
        self.spawn_piece()

    def clear_lines(self):
        lines_cleared = 0
        y = BOARD_HEIGHT - 1
        while y >= 0:
            if all(self.board[y]):
                del self.board[y]
                self.board.insert(0, [0] * BOARD_WIDTH)
                lines_cleared += 1
            else:
                y -= 1

        if lines_cleared:
            self.lines += lines_cleared
            self.score += [0, 100, 300, 500, 800][lines_cleared] * self.level
            self.level = self.lines // 10 + 1

    def move(self, dx, dy):
        if self.valid_position(self.current_x + dx, self.current_y + dy, self.current_piece):
            self.current_x += dx
            self.current_y += dy
            return True
        return False

    def rotate(self):
        rotations = SHAPES[self.current_shape]
        new_rotation = (self.current_rotation + 1) % len(rotations)
        new_piece = rotations[new_rotation]

        if self.valid_position(self.current_x, self.current_y, new_piece):
            self.current_rotation = new_rotation
            self.current_piece = new_piece
            return True
        # Wall kick attempts
        for kick in [-1, 1, -2, 2]:
            if self.valid_position(self.current_x + kick, self.current_y, new_piece):
                self.current_x += kick
                self.current_rotation = new_rotation
                self.current_piece = new_piece
                return True
        return False

    def drop(self):
        while self.move(0, 1):
            pass
        self.lock_piece()

    def step(self):
        if not self.move(0, 1):
            self.lock_piece()


def draw_board(win, game, start_y, start_x):
    # Draw border
    for y in range(BOARD_HEIGHT + 2):
        win.addstr(start_y + y, start_x, '|')
        win.addstr(start_y + y, start_x + BOARD_WIDTH * 2 + 1, '|')
    win.addstr(start_y + BOARD_HEIGHT + 1, start_x, '+' + '-' * (BOARD_WIDTH * 2) + '+')
    win.addstr(start_y, start_x, '+' + '-' * (BOARD_WIDTH * 2) + '+')

    # Draw board
    colors = [0, 1, 2, 3, 4, 5, 6, 7]
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            cell = game.board[y][x]
            char = '  ' if cell == 0 else '[]'
            color = colors[cell] if cell < len(colors) else 0
            try:
                win.addstr(start_y + y + 1, start_x + 1 + x * 2, char, curses.color_pair(color))
            except:
                win.addstr(start_y + y + 1, start_x + 1 + x * 2, char)

    # Draw current piece
    if game.current_piece and not game.game_over:
        color = SHAPE_NAMES.index(game.current_shape) + 1
        for dy, dx in game.current_piece:
            px, py = game.current_x + dx, game.current_y + dy
            if 0 <= py < BOARD_HEIGHT and 0 <= px < BOARD_WIDTH:
                try:
                    win.addstr(start_y + py + 1, start_x + 1 + px * 2, '[]', curses.color_pair(color))
                except:
                    win.addstr(start_y + py + 1, start_x + 1 + px * 2, '[]')


def draw_info(win, game, start_y, start_x):
    win.addstr(start_y, start_x, f'Score: {game.score}')
    win.addstr(start_y + 1, start_x, f'Lines: {game.lines}')
    win.addstr(start_y + 2, start_x, f'Level: {game.level}')

    win.addstr(start_y + 4, start_x, 'Next:')
    next_piece = SHAPES[game.next_shape][0]
    color = SHAPE_NAMES.index(game.next_shape) + 1
    for dy, dx in next_piece:
        try:
            win.addstr(start_y + 5 + dy, start_x + dx * 2, '[]', curses.color_pair(color))
        except:
            win.addstr(start_y + 5 + dy, start_x + dx * 2, '[]')

    win.addstr(start_y + 10, start_x, 'Controls:')
    win.addstr(start_y + 11, start_x, '← → : Move')
    win.addstr(start_y + 12, start_x, '↓   : Soft drop')
    win.addstr(start_y + 13, start_x, '↑/z : Rotate')
    win.addstr(start_y + 14, start_x, 'Space: Hard drop')
    win.addstr(start_y + 15, start_x, 'p   : Pause')
    win.addstr(start_y + 16, start_x, 'q   : Quit')


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(50)

    # Initialize colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)     # I
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # O
    curses.init_pair(3, curses.COLOR_MAGENTA, -1)  # T
    curses.init_pair(4, curses.COLOR_GREEN, -1)    # S
    curses.init_pair(5, curses.COLOR_RED, -1)      # Z
    curses.init_pair(6, curses.COLOR_BLUE, -1)     # J
    curses.init_pair(7, curses.COLOR_WHITE, -1)    # L

    game = Tetris()
    last_fall = time.time()
    paused = False

    while True:
        stdscr.clear()

        # Calculate positions
        height, width = stdscr.getmaxyx()
        board_x = (width - BOARD_WIDTH * 2 - 20) // 2
        board_y = (height - BOARD_HEIGHT - 2) // 2

        if board_x < 0:
            board_x = 0
        if board_y < 0:
            board_y = 0

        draw_board(stdscr, game, board_y, board_x)
        draw_info(stdscr, game, board_y, board_x + BOARD_WIDTH * 2 + 4)

        if game.game_over:
            msg = "GAME OVER! Press 'r' to restart or 'q' to quit"
            stdscr.addstr(board_y + BOARD_HEIGHT // 2, board_x + 1, msg[:BOARD_WIDTH*2])
        elif paused:
            stdscr.addstr(board_y + BOARD_HEIGHT // 2, board_x + 5, "PAUSED")

        stdscr.refresh()

        # Handle input
        key = stdscr.getch()

        if key == ord('q'):
            break
        elif key == ord('r'):
            game = Tetris()
            paused = False
            last_fall = time.time()
        elif key == ord('p') and not game.game_over:
            paused = not paused
        elif not paused and not game.game_over:
            if key == curses.KEY_LEFT:
                game.move(-1, 0)
            elif key == curses.KEY_RIGHT:
                game.move(1, 0)
            elif key == curses.KEY_DOWN:
                game.move(0, 1)
            elif key == curses.KEY_UP or key == ord('z'):
                game.rotate()
            elif key == ord(' '):
                game.drop()

        # Auto fall
        if not paused and not game.game_over:
            fall_speed = max(0.1, 1.0 - (game.level - 1) * 0.1)
            if time.time() - last_fall > fall_speed:
                game.step()
                last_fall = time.time()


if __name__ == '__main__':
    curses.wrapper(main)
