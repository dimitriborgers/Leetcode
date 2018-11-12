# Q355 Design Snake Game
from collections import defaultdict, deque

class SnakeGame:

    def __init__(self, width,height,food):
        self.__width = width
        self.__height = height
        self.__score = 0
        self.__food = deque(food)
        self.__snake = deque([(0, 0)])
        self.__direction = {"U": (-1, 0), "L": (0, -1), "R": (0, 1), "D": (1, 0)}
        self.__lookup = defaultdict(int)
        self.__lookup[(0, 0)] += 1

    def move(self, direction):

        def valid(x, y):
            #need lookup so that validation check done in O(1)
            return 0 <= x < self.__height and 0 <= y < self.__width and (x, y) not in self.__lookup

        d = self.__direction[direction]
        x, y = self.__snake[-1][0] + d[0], self.__snake[-1][1] + d[1]
        tail = self.__snake[0]
        self.__lookup.pop(self.__snake[0])
        self.__snake.popleft()

        if not valid(x, y):
            return -1
        elif self.__food and (self.__food[0][0], self.__food[0][1]) == (x, y):
            self.__score += 1
            self.__food.popleft()
            self.__snake.appendleft(tail)
            self.__lookup[tail] += 1

        #adds (x,y) to original deque
        self.__snake += (x, y),
        self.__lookup[(x, y)] += 1

        return self.__score

# Q375 Guess Number Higher or Lower II
class Solution:
    def getMoneyAmount(self, n):
        pay = [[0] * n for _ in range(n+1)]
        for i in reversed(range(n)):
            for j in range(i+1, n):
                pay[i][j] = min(k+1 + max(pay[i][k-1], pay[k+1][j]) for k in range(i, j+1))
        return pay[0][n-1]

# Q393 UTF-8 Validation
#0b is like 0x - it indicates the number is formatted in binary (0x indicates the number is in hex).
#bin(30)[2:].zfill(8) //if bin(30) is less than 8 digits, it fills the left side with 0s.
#bin() // returns a string
#print(1 == 0b1) //returns True
#print('{:08b}'.format(2)) //adds 0s to the left of digit inputted
#print('{:8b}'.format(2)) //adds space to the left of digit inputted
#print('{:8b}'.format(2)) will output 2 in binary
#print('{:8d}'.format(2)) will output 2 in decimal
#bin(128) //10000000 (1 + 7*0)
#bin(255) //11111111 (8*1)
#int(binary_number,2) //converts binary to decimal
#print(int(bin(a)[a.bit_length()-6:],2))  -> get 8 least sig bits
#bit_length() does not include 0b
class Solution:
    def validUtf8(self, data):
        count = 0
        for c in data:
            if count == 0:
                if (c >> 5) == 0b110:
                    count = 1
                elif (c >> 4) == 0b1110:
                    count = 2
                elif (c >> 3) == 0b11110:
                    count = 3
                elif (c >> 7):
                    return False
            else:
                if (c >> 6) != 0b10:
                    return False
                count -= 1
        return count == 0

# Q399 Evaluate Division
import collections

class Solution:
    def calcEquation(self, equations, values, query):
        def check(up, down, lookup, visited):
            if up in lookup and down in lookup[up]:
                return (True, lookup[up][down])
            for k, v in lookup[up].items():
                if k not in visited:
                    visited.add(k)
                    tmp = check(k, down, lookup, visited)
                    #this check of tmp[0] prevents the function from returning too quickly. It allows all steps in loop
                    if tmp[0]:
                        return (True, v * tmp[1])
            return (False, 0)

        lookup = collections.defaultdict(dict)
        for i, e in enumerate(equations):
            lookup[e[0]][e[1]] = values[i]
            if values[i]:
                lookup[e[1]][e[0]] = 1 / values[i]

        result = []
        for q in query:
            visited = set()
            tmp = check(q[0], q[1], lookup, visited)
            #can put a conditional statement in an append() statement
            result.append(tmp[1] if tmp[0] else -1)
        return result
