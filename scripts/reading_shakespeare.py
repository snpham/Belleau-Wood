import os

# must run on linux terminal


if __name__ == '__main__':

    os.system('head -10 datasets/shakes.txt')

    os.system('wc  datasets/shakes.txt')

    os.system(" tr -sc 'A-Za-z' '\n' < datasets/shakes.txt | head")

