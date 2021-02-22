import game
import torch
from torch import nn, optim
import numpy as np
import concurrent.futures 
from re_train import Network, cust_loss 

class Test():

    def main(self):
        self.model = Network()
        self.model.load_state_dict(torch.load('weights_model.pth'))
        self.model.eval()
        board_diameter = [25,50,100,500,1_000,10_000,100_000,1_000_000]

        for boardndx in board_diameter:
            with torch.no_grad():
                total_loss = 0
                iterations = 100
                self.boardndx = boardndx
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = [executor.submit(self.single, i) for i in range(iterations)]

                    for f in concurrent.futures.as_completed(results):
                        print(f.result())
                        total_loss += f.result()
                print('board index: ', boardndx)
                print('total loss: ',total_loss/iterations)
                to_wrt = f"{boardndx}, {total_loss/iterations}\n"
                # with open('test_result.txt', 'a') as w:
                #     w.write(to_wrt)


    def single(self, seednum):
        np.random.seed(seednum)
        cur_game = game.Board(self.boardndx)

        for j in range(cur_game.best_dist):

            if cur_game.best_dist > 10:
                debug=True

            a = cur_game.seeker_loc
            b = cur_game.goal_loc

            source = np.concatenate((a, b))
            source = source.astype('float32')
            board = torch.tensor(source, requires_grad=True)
            board = torch.flatten(board)
            board = board.view(-1, board.shape[0])

            output = self.model.forward(board)
            choice = torch.argmax(output)
            cur_game.move(int(choice))
            
        game_loss = cust_loss(cur_game)
        return game_loss

if __name__ == '__main__':
    t = Test()
    t.main()