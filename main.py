from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import njit
import random
import numpy as np
import os


def color_map():
    """creates color map"""
    new = cm.get_cmap('Blues', 256)(np.linspace(0, 1, 256))
    new[:64, :] = np.array([256/256, 256/256, 256/256, 1])
    new[64:128, :] = world.plague
    new[128:192, :] = np.array([0.03137255, 0.18823529, 0.41960784, 1])
    new[192:256, :] = np.array([0/256, 160/256, 0/256, 1])
    return ListedColormap(new)


def clear():
    """clears terminal"""
    _ = os.system('clear')


def integer(message, code, val):
    """takes integer input"""
    try:
        variable = input(message)
        clear()
        variable = int(variable)
        if code == 0:
            return variable
        elif code == 1 and variable <= 100:
            return variable
        elif code == 2 and variable <= 100 - val:
            return variable
        else:
            print('Try Again')
            return integer(message, code, val)
    except ValueError:
        print('Must be integer!')
        return integer(message, code, val)


def plague_mode():
    """asks to play plague mode"""
    clear()
    try:
        a = input('plague mode (y/n) ')
        if a == 'y':
            return 5
        elif a == 'n':
            return 4
        else:
            return plague_mode()
    except ValueError:
        return plague_mode()


class World:
    def __init__(self):
        """generates the world as an array 1-alive 0-not-alive"""
        self.nrows = integer('number of rows: ', 0, 0)
        self.ncols = integer('number of cols: ', 0, 0)
        self.games = integer('number of rounds: ', 0, 0)
        self.per_a = integer('organism a% (max 100) ', 1, 0)
        self.per_b = integer(f'organism b% (max {100 - self.per_a}) ', 2, self.per_a)
        self.per_c = integer(f'organism c% (max {100 - (self.per_a + self.per_b)}) ', 2, self.per_a + self.per_b)
        self.plague = np.array([0.12710496, 0.44018454, 0.70749712, 1])
        self.mode = plague_mode()
        self.per_empty = 100 - (self.per_a + self.per_b + self.per_c)
        random_array = np.random.choice([1, 2, 3, 0], size=(self.nrows * self.ncols), p=[self.per_a/100,
                                                                                         self.per_b/100,
                                                                                         self.per_c/100,
                                                                                         self.per_empty/100])
        self.grid = np.reshape(np.array(random_array, dtype=int), (self.nrows, self.ncols))
        self.buffer = np.zeros((self.nrows + 2, self.ncols + 2, 4), dtype=int)


@njit(nogil=True)
def update(nrows, ncols, grid, buffer, mode):
    """updates world to rules of cgol"""
    buffer *= 0
    for i in range(nrows):
        for j in range(ncols):
            buffer[i+1, j+1, 0] += grid[i, j]
    for i in range(nrows):
        for j in range(ncols):
            for k, l in [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]:
                if buffer[i+1+k, j+1+l, 0] == 1:
                    buffer[i, j, 1] += 1
                elif buffer[i+1+k, j+1+l, 0] == 2:
                    buffer[i, j, 2] += 1
                elif buffer[i+1+k, j+1+l, 0] == 3:
                    buffer[i, j, 3] += 1
    for i in range(nrows):
        for j in range(ncols):
            if buffer[i, j, 1] + buffer[i, j, 2] + buffer[i, j, 3] > 6:
                grid[i, j] = 0
            elif grid[i, j] == 1:
                if 1 < buffer[i, j, 1] < mode:
                    grid[i, j] = 1  # survives
                else:
                    grid[i, j] = 0  # dies
            # b
            elif grid[i, j] == 2:
                if 1 < buffer[i, j, 2] < 4:
                    grid[i, j] = 2  # survives
                else:
                    grid[i, j] = 0  # dies
            # c
            elif grid[i, j] == 3:
                if 1 < buffer[i, j, 3] < 4:
                    grid[i, j] = 3  # survives
                else:
                    grid[i, j] = 0  # dies
            # dead
            elif grid[i, j] == 0:
                # surrounded by 3 and 3 of one type
                if buffer[i, j, 1] == 3 and buffer[i, j, 2] == 3:
                    grid[i, j] = random.randint(1, 2)  # a xor b born
                elif buffer[i, j, 2] == 3 and buffer[i, j, 3] == 3:
                    grid[i, j] = random.randint(2, 3)  # b xor c born
                elif buffer[i, j, 3] == 3 and buffer[i, j, 1] == 3:
                    if random.randint(1, 2) == 1:
                        grid[i, j] = 1  # a born
                    else:
                        grid[i, j] = 3  # c born
                # surrounded by 3 of one type
                elif buffer[i, j, 1] == 3:
                    grid[i, j] = 1  # a born
                elif buffer[i, j, 2] == 3:
                    grid[i, j] = 2  # b born
                elif buffer[i, j, 3] == 3:
                    grid[i, j] = 3  # c born
                # surrounded by 2 of one type (1/20)
                elif buffer[i, j, 1] == 2:
                    if random.randint(1, 20) == 1:
                        grid[i, j] = 1  # a born
                    else:
                        grid[i, j] = 0  # stays dead
                elif buffer[i, j, 2] == 2:
                    if random.randint(1, 20) == 1:
                        grid[i, j] = 2  # b born
                    else:
                        grid[i, j] = 0  # stays dead
                elif buffer[i, j, 3] == 2:
                    if random.randint(1, 20) == 1:
                        grid[i, j] = 3  # c born
                    else:
                        grid[i, j] = 0  # stays dead
                # surrounded by 1 of one type (1/100)
                elif buffer[i, j, 1] == 2:
                    if random.randint(1, 100) == 1:
                        grid[i, j] = 1  # a born
                    else:
                        grid[i, j] = 0  # stays dead
                elif buffer[i, j, 2] == 2:
                    if random.randint(1, 100) == 1:
                        grid[i, j] = 2  # b born
                    else:
                        grid[i, j] = 0  # stays dead
                elif buffer[i, j, 3] == 2:
                    if random.randint(1, 100) == 1:
                        grid[i, j] = 3  # c born
                    else:
                        grid[i, j] = 0  # stays dead
                else:
                    grid[i, j] = 0  # stays dead
            else:
                grid[i, j] = 0
    return grid


def plot_world():
    """plots world"""
    hm = plt.imshow(world.grid, cmap=c_map)
    plt.draw()
    plt.pause(.02)
    hm.remove()


if __name__ == '__main__':
    clear()
    world = World()
    if world.mode == 5:
        world.plague = np.array([170/256, 0/256, 0/256, 1])
    c_map = color_map()
    plt.figure()
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    s = 0
    while s < world.games:
        world.grid = update(world.nrows, world.ncols, world.grid, world.buffer, world.mode)
        plot_world()
        s += 1
    hm = plt.imshow(world.grid, cmap=c_map)
    plt.draw()
    plt.show()
