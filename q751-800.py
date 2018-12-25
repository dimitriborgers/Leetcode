# Q753 Cracking the Safe
class Solution:
    #Hierholzer's Algorithm
    #A node plus edge represents a complete edge
    #Because the in-degrees and out-degrees of each node are equal, we can only get stuck at u, which forms a cycle.
    def crackSafe(self, n, k):
        seen = set()
        ans = []
        #To change to combinations, just make sure set isn't already in seen, this works since {1,2,3} == {1,3,2} (order does not matter)
        def dfs(node):
            #notice how you can use k here since sub-routines can access caller routine variables
            for x in map(str, range(k)):
                nei = node + x
                if nei not in seen:
                    seen.add(nei)
                    dfs(nei[1:])
                    ans.append(x)

        dfs("0" * (n-1))
        return "".join(ans) + "0" * (n-1)

# Q763 Partition Labels
from collections import defaultdict

class Solution:
    def partitionLabels(self, S):
        lookup = defaultdict(list)
        for pos,e in enumerate(S):
            if e not in lookup:
                lookup[e].append(pos)
                lookup[e].append(pos)
            else:
                lookup[e][-1] = pos

        merged = []
        for k,v in lookup.items():
            if len(merged) == 0:
                merged.append(v)
            else:
                if v[0] < merged[-1][1] and v[1] > merged[-1][0]:
                    merged[-1][1] = max(merged[-1][1],v[1])
                else:
                    merged.append(v)

        return [end-start+1 for start,end in merged]

# Q766 Toeplitz Matrix
class Solution:
    def isToeplitzMatrix(self, matrix):
        return all(i == 0 or j == 0 or matrix[i-1][j-1] == val for i, row in enumerate(matrix) for j, val in enumerate(row))

# Q774 Minimize Max Distance to Gas Station
class Solution:
    def minmaxGasDist(self, stations, K):
        def possible(D):
            return sum(int((stations[i+1] - stations[i]) / D)
                       for i in range(len(stations) - 1)) <= K

        lo, hi = 0, 10**8
        while hi - lo > 1e-6:
            mi = (lo + hi) / 2.0
            if possible(mi):
                hi = mi
            else:
                lo = mi
        return lo

# Q776 Split BST
class Solution1:
    #Recursion is a good way to be able to change cur.left and cur.right without having a parent pointer
    def splitBST(self, root, V):
        if not root:
            return [None,None]
        if root.val==V:
            a=root.right
            root.right=None
            return [root,a]
        elif root.val<V:
            small,large=self.splitBST(root.right,V)
            root.right=small
            return [root,large]
        else:
            small,large=self.splitBST(root.left,V)
            root.left=large
            return [small,root]

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.parent = None

class Solution2:
    def splitBST(self, root, V):
        if not root:
            return None,None

        cur = root
        direction = None

        while cur:
            if cur.val > V:
                if not direction:
                    direction = 'L'
                temp = cur
                cur = cur.left
                cur.parent = temp
            elif cur.val < V:
                if not direction:
                    direction = 'R'
                temp = cur
                cur = cur.right
                cur.parent = temp
            else:
                break

        if direction == 'R':
            if cur.left:
                cur.parent.right = cur.left
                cur.left = None
                return cur,root
            else:
                if cur == cur.parent.right:
                    cur.parent.right = None
                    return cur,root
                else:
                    cur.parent.parent.right = cur
                    cur.parent.left = None
                    return cur.parent,root
        else:
            if cur.right:
                cur.parent.left = cur.right
                cur.right = None
                return cur,root
            else:
                if cur == cur.parent.left:
                    cur.parent.left = None
                    return cur,root
                else:
                    cur.parent.parent.left = None
                    return cur,root

# Q777 Swap Adjacent in LR String
class Solution:
    def canTransform(self, start, end):
        #replace(x,y) replaces all instances of x with y
        if start.replace('X', '') != end.replace('X', ''):
            return False

        ctr = collections.Counter()
        for s, e in zip(start, end):
            ctr[s] += 1
            ctr[e] -= 1
            if ctr['L'] > 0 or ctr['R'] < 0:
                return False
        return True

# Q787 Cheapest Flights Within K Stops
import heapq,collections

#Two ways to use heapq for Djikstra
#1. If you add all elements to heapq at beginning, you have to run heapify everytime you change a node. Even if you change the value of a node, it will not update the composition of the heap tree. So, you must force it.
#w. If you add elements only after you've altered them, then you don't need to run heapify. The benefit of this is that you can add elements to the heap without changing the composition of the tree before you added elements.
class Solution:
    def findCheapestPrice(self, n, flights, src, dst, k):
        f = collections.defaultdict(dict)
        for a, b, p in flights:
            f[a][b] = p
        heap = [(0, src, k + 1)]
        while heap:
            p, i, k = heapq.heappop(heap)
            if i == dst:
                return p
            if k > 0:
                for j in f[i]:
                    heapq.heappush(heap, (p + f[i][j], j, k - 1))
        return -1

# Q799 Champagne Tower
class Solution:
    def champagneTower(self, poured, query_row, query_glass):
        A = [[0] * k for k in range(1, 102)]
        A[0][0] = poured
        for r in range(query_row + 1):
            for c in range(r+1):
                q = (A[r][c] - 1.0) / 2.0
                if q > 0:
                    A[r+1][c] += q
                    A[r+1][c+1] += q

        return min(1, A[query_row][query_glass])

# Q800 Similar RGB Color
class Solution:
    def similarRGB(self, color):
        def f(comp):
            #int(x,y) y represents what output type you want. 2 for binary, 10 for decimal, 16 for hex.
            q, r = divmod(int(comp, 16), 17)
            if r > 8:
                q += 1
            return '{:02x}'.format(17 * q)

        return '#' + f(color[1:3]) + f(color[3:5]) + f(color[5:])
