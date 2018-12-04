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
        return self.size[len(self.size) - 1] - 1

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

# Q815 Bus Routes
from collections import defaultdict

class Solution:
    #Using DFS, may not be quickest path
    def numBusesToDestination(self, routes, S, T):
        lookup_stops = defaultdict(list)
        lookup_buses = defaultdict(list)
        set_stops = set()
        set_buses = set()

        for r,row in enumerate(routes):
            for c,val in enumerate(row):
                lookup_stops[val].append(r)
                lookup_buses[r].append(val)

        start = lookup_stops[S]
        def dfs(buses):
            for bus in buses:
                if bus not in set_buses:
                    set_buses.add(bus)
                    for stop in lookup_buses[bus]:
                        if stop == T:
                            return (True,1)
                        if stop not in set_stops:
                            set_stops.add(stop)
                            tmp = dfs(lookup_stops[stop])
                            if tmp[0]:
                                return (True,tmp[1]+1)
            return (False,0)

        tmp = dfs(start)
        return tmp[1] if tmp[0] else -1

from collections import defaultdict,deque

class Solution1:
    #Using BFS
    def numBusesToDestination(self, routes, S, T):
        lookup_stops = defaultdict(list)
        lookup_buses = defaultdict(list)
        set_stops = set()
        set_buses = set()
        queue_stops = deque()
        queue_buses = deque()

        for r,row in enumerate(routes):
            for c,val in enumerate(row):
                lookup_stops[val].append(r)
                lookup_buses[r].append(val)

        start = lookup_stops[S]
        count = 0
        queue_buses.extend(start)

        while queue_buses:
            bus = queue_buses.popleft()
            if bus not in set_buses:
                count += 1
                set_buses.add(bus)
                queue_stops.extend(lookup_buses[bus])
                while queue_stops:
                    stop = queue_stops.popleft()
                    if stop == T:
                        return count
                    if stop not in set_stops:
                        set_stops.add(stop)
                        queue_buses.extend(lookup_stops[stop])
        return -1

# Q818 Race Car
import heapq
from collections import deque

class Solution:
    #BFS algorithm- for each position we have two choices: either accelerate or reverse.
    def racecar(self, target):
        queue = deque()
        #start at position 0 with speed 1
        queue.append((0,1))

        visited = set()
        visited.add((0,1))

        #level represents how many commands it takes to reach target
        level = 0
        while queue:
            #this limits the amount of times you run the loop (since you are adding elements to queue)
            for _ in range(len(queue)):
                cur = queue.popleft()

                if cur[0] == target:
                    return level

                #cur[0]+cur[1] -> position += speed
                #cur[1] << 1 -> speed *= 2
                nxt = (cur[0]+cur[1],cur[1] << 1)
                key = (nxt[0],nxt[1])

                #Command A
                if key not in visited and nxt[0] > 0 and nxt[0] < (target << 1):
                    queue.append(nxt)
                    visited.add(key)

                nxt = (cur[0],-1 if cur[1] > 0 else 1)
                key = (nxt[0],nxt[1])

                #Command R
                if key not in visited and nxt[0] > 0 and nxt[0] < (target << 1):
                    queue.append(nxt)
                    visited.add(key)
                queue.append(nxt)
                visited.add(key)

            level += 1
        return -1

class Solution1:
    #Using Dijkstra's Algorithm
    def racecar(self, target):
        #bit_length(): Return the number of bits necessary to represent an integer in binary, excluding the sign and leading zeros
        K = target.bit_length() + 1
        #multiplies 1 to the power of the length of K in binary
        barrier = 1 << K
        pq = [(0, target)]
        #max 2 * barrier + 1 number of nodes in pq
        dist = [float('inf')] * (2 * barrier + 1)
        dist[target] = 0

        while pq:
            steps, targ = heapq.heappop(pq)
            if dist[targ] > steps:
                continue

            for k in range(K+1):
                #multiply by k
                walk = (1 << k) - 1
                steps2, targ2 = steps + k + 1, walk - targ
                if walk == targ:
                    steps2 -= 1 #No "R" command if already exact

                if abs(targ2) <= barrier and steps2 < dist[targ2]:
                    heapq.heappush(pq, (steps2, targ2))
                    dist[targ2] = steps2

        return dist[0]

# Q819 Most Common Word
from collections import Counter
import string

class Solution:
    def mostCommonWord(self, paragraph, banned):
        count = Counter()
        exclude = set(string.punctuation)

        for punc in exclude:
            paragraph = paragraph.replace(punc,' ')

        count.update(word.lower() for word in paragraph.split(' ') if word != '')
        for pair in count.most_common():
            if pair[0] not in banned:
                return pair[0]

# Q833 Find And Replace in String
class Solution:
    def findReplaceString(self, S, indexes, sources, targets):
        original = str(S)
        updated_str = str(S)
        starting_index = offset = 0
        changes = sorted(zip(indexes,sources,targets))

        for i in changes:
            if original[i[0]:i[0]+len(i[1])] == i[1]:
                updated_str = updated_str[:i[0]+offset] + i[2] + updated_str[i[0]+len(i[1])+offset:]
                offset = offset + (len(i[2]) - len(i[1]))

        return updated_str

# Q843 Guess the Word
class Solution:
    def findSecretWord(self, wordlist, master):

        def pair_matches(a, b):         # count the number of matching characters
            return sum(c1 == c2 for c1, c2 in zip(a, b))

        def most_overlap_word():
            counts = [[0 for _ in range(26)] for _ in range(6)]     # counts[i][j] is nb of words with char j at index i
            for word in candidates:
                for i, c in enumerate(word):
                    counts[i][ord(c) - ord("a")] += 1

            best_score = 0
            for word in candidates:
                score = 0
                for i, c in enumerate(word):
                    score += counts[i][ord(c) - ord("a")]           # all words with same chars in same positions
                if score > best_score:
                    best_score = score
                    best_word = word

            return best_word

        candidates = wordlist[:]        # all remaining candidates, initially all words
        while candidates:

            s = most_overlap_word()     # guess the word that overlaps with most others
            matches = master.guess(s)

            if matches == 6:
                return

            candidates = [w for w in candidates if pair_matches(s, w) == matches]   # filter words with same matches

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
from itertools import takewhile

class Solution:
    def maxDistToClosest(self, seats):
        seq = [0]*len(seats)
        for i in range(len(seq)):
            if not seats[i]:
                seq[i] = 1

        leftEdgeList = sum(list(takewhile(lambda x: x>0,seq)))
        leftEdge = leftEdgeList if leftEdgeList else 0

        for i in range(1,len(seq)):
            if seq[i] and seq[i-1]:
                seq[i] += seq[i-1]
        if seq[-1] > max((max(seq)+1) // 2,leftEdge):
            return seq[-1]
        return max(leftEdge,(max(seq)+1) // 2)

# Q850 Rectangle Area II
import itertools,functools

#∣A∪B∪C∣=∣A∣+∣B∣+∣C∣−∣A∩B∣−∣A∩C∣−∣B∩C∣+∣A∩B∩C∣
#|𝐴∪𝐵∪𝐶∪𝐷|=(|𝐴|+|𝐵|+|𝐶|+|𝐷|)−(|𝐴∩𝐵|+|𝐴∩𝐶|+|𝐴∩𝐷|+|𝐵∩𝐶|+|𝐵∩𝐷|+|𝐶∩𝐷|)+(|𝐴∩𝐵∩𝐶|+|𝐴∩𝐵∩𝐷|+|𝐴∩𝐶∩𝐷|+|𝐵∩𝐶∩𝐷|)−(|𝐴∩𝐵∩𝐶∩𝐷|)
#odds are added, evens are removed
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
            #since rectangles is a 2D array, the combinations are combinations or the inner lists, not the items within the inner lists
            #this is how you create the ∣A∪B∪C∣=∣A∣+∣B∣+∣C∣−∣A∩B∣−∣A∩C∣−∣B∩C∣+∣A∩B∩C∣
            for group in itertools.combinations(rectangles, size):
                #(-1) ** (size + 1) decides if you're adding or subtracting
                ans += (-1) ** (size + 1) * area(functools.reduce(intersect, group))

        #return it modulo 10^9 + 7
        return ans % (10**9 + 7)
