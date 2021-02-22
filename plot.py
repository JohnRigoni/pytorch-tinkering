import matplotlib.pyplot as plt
import numpy as np


def show():
    with open("net_result.txt") as a:
        nums = a.read().split(',')[:-1]

    for i,x in enumerate(nums):
        nums[i] = float(x)

    x = np.array([w for w in range(len(nums))])
    y = np.array(nums) / 10

    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef) 

    plt.plot(x, y, 'y', x, poly1d_fn(x))
    plt.show()


if __name__ == "__main__":
    show()