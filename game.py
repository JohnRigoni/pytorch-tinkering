import numpy as np
import copy
from math import floor
np.random.seed(100)

class Board():
    def __init__(self, board_diameter, train=False):
        self.board_diameter = board_diameter
        seeker_loc_init = floor(board_diameter/2)

        # self.board = np.zeros((board_diameter,board_diameter), dtype=np.float32)
        self.seeker_loc = np.array([seeker_loc_init,seeker_loc_init])
        self.seeker_loc_prev = np.array([seeker_loc_init,seeker_loc_init])
        # self.board[tuple(self.seeker_loc)] = 2
        if not train:
            x = np.random.randint(0,board_diameter)
            y = np.random.randint(0,board_diameter)
            while x == seeker_loc_init and y == seeker_loc_init:
                x = np.random.randint(0,board_diameter)
                y = np.random.randint(0,board_diameter)
        else:
            bound = board_diameter - 1
            sel = [(0,0), (bound,0), (0,bound), (bound,bound)]
            x,y = sel[np.random.randint(0,4)]

        self.goal_loc =  np.array([x,y])
        # self.board[tuple(self.goal_loc)] = 1

        self.state = 0
        self.best_dist = self.seeker_loc - self.goal_loc
        self.best_dist = np.absolute(self.best_dist)
        self.best_dist = np.sum(self.best_dist)
        self.target_calc()


    def move(self, direction):
        self.seeker_loc_prev = copy.deepcopy(self.seeker_loc)
        if direction == 0:
            if self.seeker_loc[0] - 1 >= 0:
                self.seeker_loc[0] -= 1
        elif direction == 1:
            if self.seeker_loc[1] + 1 <= self.board_diameter - 1:
                self.seeker_loc[1] += 1
        elif direction == 2:
            if self.seeker_loc[0] + 1 <= self.board_diameter - 1:
                self.seeker_loc[0] += 1
        elif direction == 3:
            if self.seeker_loc[1] - 1 >= 0:
                self.seeker_loc[1] -= 1
        # self.board[tuple(self.seeker_loc_prev)] = 0
        # self.board[tuple(self.seeker_loc)] = 2
        if np.array_equal(self.seeker_loc, self.goal_loc):
            self.state = 1

    def target_calc(self):
        self.move_list = []
        vert = self.seeker_loc[0] - self.goal_loc[0]
        hor = self.seeker_loc[1] - self.goal_loc[1]

        if vert >= 1:
            # move up
            self.move_list.append(0)
        elif vert <= -1:
            # move down
            self.move_list.append(2)
        if hor <= -1:
            # move right
            self.move_list.append(1)
        elif hor >= 1:
            # move left
            self.move_list.append(3)
        
    def target_get(self):
        self.target_calc()
        if len(self.move_list) > 1:
            len(self.move_list)
            return self.move_list[np.random.randint(0,2)]
        else:
            return self.move_list[0]

    def p(self):
        for uti in self.board:
            print(uti)
        print()

if __name__ == "__main__":
    b = Board(board_diameter=5)
    # b.p()
    import readchar

    move_count = 0
    while b.state == 0 and move_count < b.best_dist:
        word = repr(readchar.readchar())
        if word == "'w'":
            b.move(0)
        elif word == "'d'":
            b.move(1)
        elif word == "'s'":
            b.move(2)
        elif word == "'a'":
            b.move(3)
        b.p()
        move_count+=1
        b.target_calc()
                
    if move_count >= b.best_dist:
        loss = b.seeker_loc - b.goal_loc
        loss = np.absolute(loss)
        loss = np.sum(loss)
        loss = loss / b.best_dist * 1/2
        
    else:
        loss = 0

    print(loss)
