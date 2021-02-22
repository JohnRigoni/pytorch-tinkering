import matplotlib.pyplot as plt
import csv

with open('test_result.txt', 'r') as r:
    data = csv.reader(r)
    x = []
    y = []
    for i in data:
        x.append(i[0])
        y.append(float(i[1]))
    
    plt.xlabel('Board Diameter')
    plt.ylabel('Custom Loss')    
    plt.plot(x,y)
    
    plt.show()