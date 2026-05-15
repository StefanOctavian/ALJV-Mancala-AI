import numpy as np
from constants import *

class MancalaEnv:
    def __init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        # 6 pits each + 2 mancalas
        self.board = [4] * 6 + [0] + [4] * 6 + [0]
        self.current_player = AGENT
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return np.array(self.board, dtype=np.float32) / 48.0

    def valid_moves(self) -> list:
        if self.current_player == AGENT:
            return [i for i in range(6) if self.board[i] > 0]
        else:
            return [i - 7 for i in range(7, 13) if self.board[i] > 0]
        
    def own_pit(self, index) -> bool:
        return (self.current_player == AGENT and 0 <= index < 6) or \
               (self.current_player == OPPONENT and 7 <= index < 13)
    
    def own_mancala(self) -> int:
        return 6 if self.current_player == AGENT else 13

    def opponent_mancala(self) -> int:
        return 13 if self.current_player == AGENT else 6

    def step(self, pit) -> tuple[np.ndarray, float, bool]:
        stones = self.board[pit]
        self.board[pit] = 0
        i = pit
        reward = 0

        # distribute stones
        while stones > 0:
            i = (i + 1) % 14
            # skip opponent's mancala
            if i == self.opponent_mancala():
                continue
            elif i == self.own_mancala() and self.current_player == AGENT:
                reward += 0.05
            self.board[i] += 1
            stones -= 1
        last_pit = i

        # check bad move: leaving yourself vulnerable to an obvious capture
        if self.current_player == AGENT:
            for j in range(7, 11):
                falls_in_pit = (j + self.board[j]) % 14
                if self.board[falls_in_pit] == 0 and self.board[12 - falls_in_pit] > 0 \
                   and not self.own_pit(falls_in_pit):
                    reward -= 0.1

        # capture rule
        if self.own_pit(last_pit) and last_pit != self.own_mancala() and self.board[last_pit] == 1:
            opposite_index = 12 - last_pit
            if self.board[opposite_index] > 0:
                captured = self.board[opposite_index] + 1
                self.board[self.own_mancala()] += captured
                self.board[opposite_index] = 0
                self.board[last_pit] = 0
                if self.current_player == AGENT:
                    reward += 0.2 * captured
                else:
                    reward -= 0.15 * captured

        done = False

        # terminal condition
        if sum(self.board[:6]) == 0 or sum(self.board[7:13]) == 0:
            done = True
            if self.board[6] > self.board[13]:
                reward = 1
            elif self.board[6] < self.board[13]:
                reward = -1
            else:
                reward = 0.1

        # extra turn rule or switch player
        if last_pit != self.own_mancala():
            self.current_player = 1 - self.current_player
        else:
            if self.current_player == AGENT:
                reward += 0.3
            else:
                reward -= 0.2

        return self.get_state(), reward, done
    
    def valid_action_mask(self):
        mask = np.zeros(6, dtype=np.float32)

        if self.current_player == 0:
            for i in range(6):
                mask[i] = 1.0 if self.board[i] > 0 else 0.0
        else:
            for i in range(7, 13):
                mask[i - 7] = 1.0 if self.board[i] > 0 else 0.0

        return mask