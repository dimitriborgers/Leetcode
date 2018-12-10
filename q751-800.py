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
