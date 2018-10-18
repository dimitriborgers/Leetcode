# Q803 Bricks Falling When Hit
class DSU:
    def __init__(self, R, C):
        #R * C is the source, and isn't a grid square
        self.par = range(R*C + 1)
        self.rnk = [0] * (R*C + 1)
        self.sz = [1] * (R*C + 1)

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr: return
        if self.rnk[xr] < self.rnk[yr]:
            xr, yr = yr, xr
        if self.rnk[xr] == self.rnk[yr]:
            self.rnk[xr] += 1

        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]

    def size(self, x):
        return self.sz[self.find(x)]

    def top(self):
        # Size of component at ephemeral "source" node at index R*C,
        # minus 1 to not count the source itself in the size
        return self.size(len(self.sz) - 1) - 1

class Solution:
    def hitBricks(self, grid, hits):
        R, C = len(grid), len(grid[0])
        def index(r, c):
            return r * C + c

        def neighbors(r, c):
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= nr < R and 0 <= nc < C:
                    yield nr, nc

        A = [row[:] for row in grid]
        for i, j in hits:
            A[i][j] = 0

        dsu = DSU(R, C)
        for r, row in enumerate(A):
            for c, val in enumerate(row):
                if val:
                    i = index(r, c)
                    if r == 0:
                        dsu.union(i, R*C)
                    if r and A[r-1][c]:
                        dsu.union(i, index(r-1, c))
                    if c and A[r][c-1]:
                        dsu.union(i, index(r, c-1))

        ans = []
        for r, c in reversed(hits):
            pre_roof = dsu.top()
            if grid[r][c] == 0:
                ans.append(0)
            else:
                i = index(r, c)
                for nr, nc in neighbors(r, c):
                    if A[nr][nc]:
                        dsu.union(i, index(nr, nc))
                if r == 0:
                    dsu.union(i, R*C)
                A[r][c] = 1
                ans.append(max(0, dsu.top() - pre_roof - 1))
        return ans[::-1]

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
            pool = [ i for i in pool if num_of_same(word,i) == num_same]
        return -1

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
