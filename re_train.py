import game
import torch
from torch import nn, optim
import numpy as np
import plot as myplot
np.random.seed(0)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden1 = nn.Linear(4, 6)
        self.hidden2 = nn.Linear(6, 6)
        self.output = nn.Linear(6, 4)
        
        # Define relu activation and softmax output 
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.relu(x)

        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x

def cust_loss(game):
        if game.state == 0:
            loss = game.seeker_loc - game.goal_loc
            loss = np.absolute(loss)
            loss = np.sum(loss)
            loss = loss / game.best_dist * 1/2      
        else:
            loss = 0       
        return np.float(loss)

def train():
    
    epochs = 400
    model = Network()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for e in range(epochs):
        
        game_batch = 8
        running_loss = 0
        optimizer.zero_grad()
        game_batch_loss = 0

        sourceL = []
        targetL = []
        for i in range(game_batch):
        
            cur_game = game.Board(board_diameter=7, train=True)
            noise_batch = 5
            for j in range(cur_game.best_dist):
                source_temp = []
                for x in range(noise_batch):

                    r1 = np.random.randint(-25,25)
                    r2 = np.random.randint(-25,25)

                    a = cur_game.seeker_loc + [r1,r2]
                    b = cur_game.goal_loc + [r1,r2]
                    
                    pos = np.concatenate((a, b))
                    pos = pos.astype('float32')

                    source_temp.append(pos)
                    targetL.append(cur_game.target_get())

                board = torch.tensor(source_temp, requires_grad=True, device=device)

                output = model.forward(board)
                for o in output:
                    sourceL.append(o)
                choice = torch.argmax(output, dim=1)
                cur_game.move(int(choice[np.random.randint(0, len(choice))]))


            game_loss = cust_loss(cur_game)
            game_batch_loss += game_loss
            print(game_loss)


        target = torch.tensor(targetL, device=device)
        sourced = torch.stack(sourceL)
        loss = loss_func(sourced, target)
        loss.backward()
        optimizer.step()


        # with open('net_result.txt', 'a') as w:
        #     w.writelines(f"{round(game_batch_loss, 4)},")
            
    # torch.save(model.state_dict(), 'weights_model.pth')


if __name__ == "__main__":
    # clear file
    with open('net_result.txt', 'w') as w:
        pass

    train()

    # myplot.show()