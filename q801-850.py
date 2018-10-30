# Q803 Bricks Falling When Hit
class DSU:
    def __init__(self, R, C):
        #R * C is the source, and isn't a grid square
        #self.parent has to be list because range() does not support item assignment, which is needed in union
        self.parent = [i for i in range(R*C + 1)]
        self.rank = [0]*(R*C + 1)
        self.size = [1]*(R*C + 1)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return
        #Same as normal DSU. In this case, you always make xr parent, so if it isn't actually bigger, than just switch it with yr
        if self.rank[xr] < self.rank[yr]:
            xr, yr = yr, xr
        if self.rank[xr] == self.rank[yr]:
            self.rank[xr] += 1

        self.parent[yr] = xr
        self.size[xr] += self.size[yr]

    def size(self, x):
        return self.size[self.find(x)]

    def top(self):
        # Size of component at ephemeral "source" node at index R*C, minus 1 to not count the source itself in the size
        # All nodes on the top edge are connected to the source node
        return self.size(len(self.size) - 1) - 1

class Solution:
    def hitBricks(self, grid, hits):
        R, C = len(grid), len(grid[0])

        #converting grid location to array location
        def index(r, c):
            return r * C + c

        #finding all neighbors of a cell in grid
        #good example of a generator, allows you to for loop through it
        #very useful help function if need to check all valid neighbors
        def neighbors(r, c):
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= nr < R and 0 <= nc < C:
                    yield nr, nc

        #In this stuation, row[:] acts as list(row)
        #You could also do this by doing a deepcopy of grid
        A = [row[:] for row in grid]
        for i, j in hits:
            A[i][j] = 0

        dsu = DSU(R, C)

        #great way of getting both index and value at same time
        for r, row in enumerate(A):
            for c, val in enumerate(row):
                if val:
                    i = index(r, c)
                    if r == 0:
                        #This is why range has to be R*C+1 in DSU init
                        #There is an ephemeral source node
                        dsu.union(R*C,i)
                    if r and A[r-1][c]:
                        dsu.union(i, index(r-1, c))
                    if c and A[r][c-1]:
                        dsu.union(i, index(r, c-1))

        ans = []
        for r, c in reversed(hits):
            pre_roof = dsu.top()
            #If original grid[r][c] was always 0, then we couldn't have had a meaningful cut - the number of dropped bricks is 0
            if grid[r][c] == 0:
                ans.append(0)
            else:
                #Otherwise, we'll look at the size of the new roof after adding this brick at (r, c), and compare them to find the number of dropped bricks
                i = index(r, c)
                for nr, nc in neighbors(r, c):
                    if A[nr][nc]:
                        dsu.union(i, index(nr, nc))
                if r == 0:
                    dsu.union(R*C,i)
                A[r][c] = 1
                ans.append(max(0, dsu.top() - pre_roof - 1))
        return ans[::-1]

# Q833 Find And Replace in String
class Solution:
    def findReplaceString(self, S, indexes, sources, targets):
        original = str(S)
        updated_str = str(S)
        starting_index = 0

        for i in zip(sources,indexes,targets):
            if original[i[1]:i[1]+len(i[0])] == i[0]:
                updated_str = updated_str[:i[1]+starting_index] + i[2] + updated_str[i[1]+len(i[0])+starting_index:]

                starting_index += len(i[2])-1

        return updated_str

# Q843 Guess the Word
import random

#class Master(object):
#    def guess(self, word):
#        """
#        :type word: str
#        :rtype int
#        """

class Solution:
    def findSecretWord(self, wordlist, master):

        def num_of_same(aa, bb):
            return sum([a == b for a, b in zip(aa,bb)])

        random.shuffle(wordlist)
        pool = list(word)

        while pool:
            word = pool.pop()
            num_same = master.guess(word)
            if num_same == len(word):
                return word
            pool = [i for i in pool if num_of_same(word,i) == num_same]
        return -1

# Q844 Backspace string Compare
class Solution:
    def backspaceCompare(self, S, T):

        def strip(word):
            count,result = 0,''
            for i in range(len(word)-1,-1,-1):
                if word[i] == '#':
                    count += 1
                else:
                    if count:
                        count -= 1
                    else:
                        result += word[i]
            return result[::-1]

        return (strip(S) == strip(T))

# Q846 Hand of Straights
class Solution:
    def isNStraightHand(self, hand, W):
        if len(hand) % W:
            return False

        result = [[] for i in range(W)]
        hand.sort()
        while hand:
            for i in result:
                i.append(hand[0])
                tmp = hand[0]
                del hand[hand.index(tmp)]
                while len(i) < W:
                    if tmp + 1 in hand:
                        i.append(tmp+1)
                        del hand[hand.index(tmp+1)]
                        tmp += 1
                    else:
                        return False

        return result

# Q849 Maximize Distance to Closest Person
class Solution:
    def maxDistToClosest(self, seats):
        seq = [0]*len(seats)
        for i in range(len(seq)):
            if not seats[i]:
                seq[i] = 1
        for i in range(1,len(seq)):
            if seq[i] and seq[i-1]:
                seq[i] += seq[i-1]
        if seq[-1] > (max(seq)+1) // 2:
            return seq[-1]
        return (max(seq)+1) // 2

# Q850 Rectangle Area II
import itertools,functools

class Solution:
    def rectangleArea(self, rectangles):
        def intersect(rec1, rec2):
            return [max(rec1[0], rec2[0]),
                    max(rec1[1], rec2[1]),
                    min(rec1[2], rec2[2]),
                    min(rec1[3], rec2[3])]

        def area(rec):
            dx = max(0, rec[2] - rec[0])
            dy = max(0, rec[3] - rec[1])
            return dx * dy

        ans = 0
        for size in range(1, len(rectangles) + 1):
            for group in itertools.combinations(rectangles, size):
                ans += (-1) ** (size + 1) * area(functools.reduce(intersect, group))

        return ans % (10**9 + 7)
