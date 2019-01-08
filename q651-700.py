# Q652 Find Duplicate Subtrees
class Solution:
    def findDuplicateSubtrees(self, root):
        def trv(node):
            if not node:
                return "null"
            struct = "%s,%s,%s" % (str(node.val), trv(node.left), trv(node.right))
            nodes[struct].append(node)
            return struct

        nodes = collections.defaultdict(list)
        trv(root)
        return [nodes[struct][0] for struct in nodes if len(nodes[struct]) > 1]

# Q653 Two Sum IV - Input is a BST
class Solution:
    def findTarget(self, root, k):

        def inorder_traversal(node):
            if node.left:
                tmp = inorder_traversal(node.left)
                if tmp:
                    return True

            if node.val in lookup:
                return True
            else:
                lookup[k-node.val] = node.val

            if node.right:
                tmp = inorder_traversal(node.right)
                if tmp:
                    return True

        lookup = {}
        result = inorder_traversal(root)
        return result if result == True else False

# Q658 Find K Closest Elements
import bisect

class Solution:
    def findClosestElements(self, arr, k, x):
        location = bisect.bisect_left(arr,x)
        bisect.insort(arr,x)
        i = location - 1
        j = location + 1
        result = []
        while k > 0:
            if 0 <= i < len(arr) and 0 <= j < len(arr):
                if abs(x - arr[i]) <= abs(x-arr[j]):
                    result.append(arr[i])
                    k -= 1
                    i -= 1
                else:
                    result.append(arr[j])
                    k -= 1
                    j += 1
            elif 0 <= i < len(arr):
                result.append(arr[i])
                k -= 1
                i -= 1
            else:
                result.append(arr[j])
                k -= 1
                j += 1
        return sorted(result)

# Q675 Cut Off Trees for Golf Event
# Time Limit Exceeded: Simple BFS
class Solution:
    def cutOffTree(self, forest):
        if forest[0][0] == 0:
            return -1

        #To loop through a 2D array using list comprehension, think of it as going through the outer, then inner loop
        ordered_trees = sorted([num for row in forest for num in row if num > 1])

        if forest[0][0] == ordered_trees[0]:
            forest[0][0] = 1
            ordered_trees = ordered_trees[1:]

        g_count = 0
        location = (0,0)
        directions = [(1,0),(-1,0),(0,1),(0,-1)]

        for tree in ordered_trees:
            l_count = 0
            level = [location]
            visited = {location}
            found = False

            while level and not found:
                next_level = []
                for l in level:
                    row = l[0]
                    col = l[1]

                    if forest[row][col] == tree:
                        g_count += l_count
                        location = (row,col)
                        forest[row][col] = 1
                        found = True
                        break

                    for d in directions:
                        if 0 <= row+d[0] < len(forest) and 0 <= col+d[1] < len(forest[0]) and forest[row+d[0]][col+d[1]] > 0 and (row+d[0],col+d[1]) not in visited:
                            #Always add to visited when first seeing it, not after it's already been added to level queue
                            visited.add((row+d[0],col+d[1]))
                            next_level.append((row+d[0],col+d[1]))

                level = next_level
                l_count += 1

            if not found:
                return -1

        return g_count

# Hadlock's algorithm
class Solution:
    def cutOffTree(self, forest):

        def hadlocks(forest, sr, sc, tr, tc):
            R, C = len(forest), len(forest[0])
            processed = set()
            deque = collections.deque([(0, sr, sc)])
            while deque:
                detours, r, c = deque.popleft()
                if (r, c) not in processed:
                    processed.add((r, c))
                    if r == tr and c == tc:
                        return abs(sr-tr) + abs(sc-tc) + 2*detours
                    for nr, nc, closer in ((r-1, c, r > tr), (r+1, c, r < tr),
                                           (r, c-1, c > tc), (r, c+1, c < tc)):
                        if 0 <= nr < R and 0 <= nc < C and forest[nr][nc]:
                            if closer:
                                deque.appendleft((detours, nr, nc))
                            else:
                                deque.append((detours+1, nr, nc))
            return -1

        trees = sorted((v, r, c) for r, row in enumerate(forest)
                       for c, v in enumerate(row) if v > 1)
        sr = sc = ans = 0
        for _, tr, tc in trees:
            d = hadlocks(forest, sr, sc, tr, tc)
            if d < 0:
                return -1
            ans += d
            sr, sc = tr, tc
        return ans



# Q679 24 Game
#self-made classes naturally have __hash__ function made for them
#heapq methods don't work with self-made classes, unless you define le,lt,etc.
from operator import truediv, mul, add, sub

class Solution:
    def judgePoint24(self, A):
        if not A:
            return False
        if len(A) == 1:
            return abs(A[0] - 24) < 1e-6

        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    B = [A[k] for k in range(len(A)) if i != k != j]
                    for op in (truediv, mul, add, sub):
                        #Makes sure you don't divide by 0
                        if op is not truediv or A[j]:
                            if (op is add or op is mul) and j > i:
                                continue
                            if op is not truediv or A[j]:
                                B.append(op(A[i], A[j]))
                                if self.judgePoint24(B):
                                    return True
                                B.pop()
        return False

# Q681 Next Closest Time
class Solution:
    def nextClosestTime(self, time):
        #cannot do [::-1] or anything of that sort in set(). It is not suscriptable
        time = [int(i) for i in time if i != ':']

        for j in range(time[-1]+1,10):
            if j in time and j != time[-1]:
                time[-1] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

        for j in range(time[-2]+1,6):
            if j in time and j != time[-2]:
                time[-2] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

        for j in range(time[-3]+1,5):
            if j in time and j != time[-3]:
                time[-3] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

        for j in range(time[-4]+1,3):
            if j in time and j != time[-4]:
                time[-4] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

        minimum = min(time)
        time = [minimum]*4
        time = ''.join(str(i) for i in time)
        #concatenation
        return time[:2]+':'+time[2:]

    def nextClosestTime2(self, time):
        cur = 60 * int(time[:2]) + int(time[3:])
        allowed = {int(x) for x in time if x != ':'}
        while True:
            cur = (cur + 1) % (24 * 60)
            #automatic packing if multiple values assigned to one variable
            #second forloop done first
            if all(digit in allowed for block in divmod(cur, 60) for digit in divmod(block, 10)):
                #2 puts two spaces of padding minimum and 0 puts a 0 in front of the number if only 1 is present
                #d is just for signed integer decimal
                #to unpack, you must put the * before the variable
                return "{:02d}:{:02d}".format(*divmod(cur, 60))

# Q682


# Q683 K Empty Slots
import bisect

class Solution:
    def kEmptySlots(self, flowers, k):
        days = [0] * len(flowers)
        #enumerate only works in left to right order
        #you cannot use it with reversed(flowers) or flowers[::-1]
        for day, position in enumerate(flowers, 1):
            days[position - 1] = day

        ans = float('inf')
        left, right = 0, k+1
        while right < len(days):
            for i in range(left + 1, right):
                if days[i] < days[left] or days[i] < days[right]:
                    left, right = i, i+k+1
                    break
            #else statement will occur after end of loop. Will not occur if break statement is encountered.
            else:
                ans = min(ans, max(days[left], days[right]))
                left, right = right, right+k+1

        return ans if ans < float('inf') else -1

    def kEmptySlots2(self, flowers, k):
        active = []
        #second arg in enumerate lets you choose what number to start enumeration on.
        for day, flower in enumerate(flowers, 1):
            #bisect.bisect tells you where exactly the value would be inserted
            i = bisect.bisect(active, flower)
            #i-(i>0) -> i>0 is 1 if true, 0 if not
            '''
            seq = []
            for i in seq[2:4]
            This does not give a out of bounds error. It will just skip this loop if the range given does not apply
            To start loop, the first number must be accurate. Loop exits if reached end or number no longer accurate
            '''
            for neighbor in active[i-(i>0):i+1]:
                if abs(neighbor - flower) - 1 == k:
                    return day
            #bisect.insort knows where to insert element
            #moves element previous at position to the right
            bisect.insort(active,flower)
        return -1

# Q684 Redundant Connection
import collections

class Solution1:
    def findRedundantConnection(self, edges):
        graph = collections.defaultdict(set)

        def dfs(source, target):
            if source not in seen:
                seen.add(source)
                if source == target:
                    return True
                return any(dfs(nei, target) for nei in graph[source])

        for u, v in edges:
            #notice that seen does not have to be self.seen since dfs is accessing an already created public element
            seen = set()
            if u in graph and v in graph and dfs(u, v):
                return u, v
            graph[u].add(v)
            graph[v].add(u)

class Solution2:
    def findRedundantConnection(self, edges):
        dsu = DSU()
        for edge in edges:
            if not dsu.union(*edge):
                return edge

class DSU:
    def __init__(self):
        self.parent = [i for i in range(1001)]
        self.rank = [0] * 1001

    def find(self, x):
        if self.parent[x] != x:
            #path compression
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    #union by rank
    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        elif self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True

# Q685 Redundant Connection II
class Solution:
    def findRedundantDirectedConnection(self, edges):
        candidate =[]
        self.par = [0]*(len(edges)+1)

        for u,v in edges:
            if self.par[v] != 0:
                candidate.append([self.par[v],v])
                candidate.append([u,v])
                break
            else:
                self.par[v] = u
        self.par = [i for i in range(len(edges)+1)]
        for u,v in edges:
            if candidate and [u,v] == candidate[1]:
                continue
            if self.find(u) == v:
                if candidate:
                    return candidate[0]
                return [u,v]
            self.par[v] = u
        return candidate[1]

    def find(self,u):
        if self.par[u] != u:
            self.par[u] = self.find(self.par[u])
        return self.par[u]


# Q686 Repeated String Match
class Solution:
    def repeatedStringMatch(self, A, B):
        q = ((len(B)-1) // len(A)) + 1
        for i in range(2):
            if B in A * (q+i):
                return q+i
        return -1

# Q692 Top K Frequent Words
from collections import Counter

class Solution1:
    #there is no built-in way to sort on two elements with one ordering being reverse
    #however, if one element is an integer, just sort them on negative ints
    def topKFrequent(self, words, k):
        count = [(k,v) for k,v in collections.Counter(words).items()]
        return [k for k,v in sorted(count,key = lambda x:(-x[1],x[0]))][:k]

#heap
class Solution2:
    def topKFrequent(self, words, k):
        count = collections.Counter(words)
        heap = [(-freq, word) for word, freq in count.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in xrange(k)]

# Q694 Number of Distinct Islands
class Solution:
    def numDistinctIslands(self, grid):
        seen = set()
        #Instead of first creating the islands and then looping through them to get their shapes, explore does both at the same time.
        def explore(r, c, r0, c0):
            if (0 <= r < len(grid) and 0 <= c < len(grid[0]) and
                    grid[r][c] and (r, c) not in seen):
                seen.add((r, c))
                shape.add((r - r0, c - c0))
                explore(r+1, c, r0, c0)
                explore(r-1, c, r0, c0)
                explore(r, c+1, r0, c0)
                explore(r, c-1, r0, c0)

        shapes = set()
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                shape = set()
                explore(r, c, r, c)
                if shape:
                    shapes.add(frozenset(shape))
        return len(shapes)

# Q695 Max Area of Island
class Solution:
    def maxAreaOfIsland(self, grid):
        m, n = len(grid), len(grid[0])

        def dfs(i, j):
            if 0 <= i < m and 0 <= j < n and grid[i][j]:
                grid[i][j] = 0
                return 1 + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i + 1, j) + dfs(i, j - 1)
            return 0

        areas = [dfs(i, j) for i in range(m) for j in range(n) if grid[i][j]]
        return max(areas) if areas else 0
