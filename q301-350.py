# Q303 Range Sum Query - Immutable
class NumArray:
    def __init__(self, nums):
        self.accu = [0]
        #You can do more that just assignment statements in __init__
        for num in nums:
            self.accu.append(self.accu[-1] + num)

    def sumRange(self, i, j):
        return self.accu[j + 1] - self.accu[i]

# Q307 Range Sum Query = Mutable
import itertools

class NumArray:

    def __init__(self, nums):
        self.original = nums
        self.nums = [0] + list(itertools.accumulate(nums))

    def update(self, i, val):
        tmp = self.original[i] - val
        for j in range(i+1,len(self.nums)):
            self.nums[j] -= tmp

    def sumRange(self, i, j):
        return self.nums[j+1]-self.nums[i]

# Q308 Range Sum Query 2D - Mutable
from itertools import accumulate

class NumMatrix:
    def __init__(self, matrix):
        self.d = [list(accumulate(row)) for row in matrix]

    def update(self, row, col, val):
        row = self.d[row]
        orig = row[col] - (row[col-1] if col else 0)
        for i in range(col, len(row)):
            row[i] += val - orig

    def sumRegion(self, row1, col1, row2, col2):
        out = 0
        for i in range(row1, row2+1):
            out += self.d[i][col2] - (self.d[i][col1-1] if col1 else 0)
        return out

# Q314 Binary Tree Vertical Order Traversal
class Solution:
    def verticalOrder(self, root):
        cols = collections.defaultdict(list)
        queue = [(root, 0)]
        for node, i in queue:
            if node:
                cols[i].append(node.val)
                #this extends the original list
                #allows you to extend multiple things at once
                queue += (node.left, i - 1), (node.right, i + 1)
        #you can sort a dictionary by its keys
        return [cols[i] for i in sorted(cols)]

# Q315 Count of Smaller Numbers After Self
from bisect import bisect_left,insort

class Solution:
    def countSmaller(self, nums):
        sort = []
        output = []
        for i in reversed(nums):
            loc = bisect_left(sort,i)
            output.append(loc)
            insort(sort,i)

        return output[::-1]

# Q317 Shortest Distance from All Buildings
# Doesn't work
class Solution1:
    def shortestDistance(self, grid):
        def check(row,col,visited):
            return True if (0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != 2 and (row,col) not in visited) else False

        def bfs(row,col,houses,distance=0):
            visited = set()
            path = []
            level = [(row,col)]

            while level:
                next_level = []
                for r,c in level:
                    if check(r,c,visited) and grid[r][c] == 1:
                        path.append(distance)
                        visited.add((r,c))
                        continue
                    visited.add((r,c))
                    if check(r+1,c,visited):
                        next_level.append((r+1,c))
                    if check(r-1,c,visited):
                        next_level.append((r-1,c))
                    if check(r,c+1,visited):
                        next_level.append((r,c+1))
                    if check(r,c-1,visited):
                        next_level.append((r,c-1))
                distance += 1
                level = next_level
            return path if len(path) == houses else [float('inf')]

        if not grid or not grid[0]:
            return -1
        minLength = float('inf')
        total_houses = 0
        for row,seq in enumerate(grid):
            for col,e in enumerate(seq):
                if grid[row][col] == 1:
                    total_houses += 1
        for row,seq in enumerate(grid):
            for col,e in enumerate(seq):
                if not grid[row][col]:
                    minLength = min(minLength,sum(bfs(row,col,total_houses)))

        return minLength if (minLength != float('inf') and minLength != 0) else -1

class Solution2:
    def shortestDistance(self, grid):
        h = len(grid)
        w = len(grid[0])

        distance = [[0 for _ in range(w)] for _ in range(h)]
        reach = [[0 for _ in range(w)] for _ in range(h)]

        buildingNum = 0

        for i in range(h):
            for j in range(w):
                if grid[i][j] == 1:
                    buildingNum += 1
                    q = [(i, j, 0)]

                    isVisited = [[False for _ in range(w)] for _ in range(h)]

                    for y, x, d in q:
                        for dy, dx in (-1, 0), (1, 0), (0, -1), (0, 1):
                            r = y + dy
                            c = x + dx

                            if 0 <= r < h and 0 <= c < w and grid[r][c] == 0 and not isVisited[r][c]:
                                distance[r][c] += d + 1
                                reach[r][c] += 1

                                isVisited[r][c] = True
                                q.append((r, c, d + 1))

        shortest = float("inf")
        for i in range(h):
            for j in range(w):
                if grid[i][j] == 0 and reach[i][j] == buildingNum:
                    shortest = min(shortest, distance[i][j])

        if shortest < float("inf"):
            return shortest
        else:
            return -1

# Q322 Coin Change
class Solution:
    def coinChange(self, coins, amount):
        dp = [0] + [float('inf')] * amount

        for i in range(1, amount + 1):
            dp[i] = min([dp[i - coin] if i - coin >= 0 else float('inf') for coin in coins]) + 1

        #[dp[amount], -1][dp[amount] == max_val] => [dp[amount], -1][0 or 1]
        #if dp[amount] == max_val:
            #[dp[amount], -1][1] => -1
        #else:
            #[dp[amount], -1][0] => dp[amount]
        return [dp[amount], -1][dp[amount] == float('inf')]

# Q323 Number of Connected Components in an Undirected Graph
class Solution:
    def countComponents(self, n, edges):
        graph = {i:[] for i in range(n)}
        for v1,v2 in edges:
            graph[v1].append(v2)
            graph[v2].append(v1)

        def dfs(node):
            for adjacent in graph[node]:
                if adjacent not in visited:
                    visited.add(adjacent)
                    paths[adjacent] = node
                    dfs(adjacent)

        paths = {}
        visited = set()
        for vertex in graph:
            if vertex not in visited:
                paths[vertex] = None
                visited.add(vertex)
                dfs(vertex)

        return sum([1 for i in paths.values() if i == None])

# Q325 Maximum Size Subarray Sum Equals k
class Solution:
    def maxSubArrayLen(self, nums, k):
        sums = {}
        cur_sum, max_len = 0, 0
        for i in range(len(nums)):
            cur_sum += nums[i]
            if cur_sum == k:
                max_len = i + 1
            elif cur_sum - k in sums:
                max_len = max(max_len, i - sums[cur_sum - k])
            if cur_sum not in sums:
                sums[cur_sum] = i
        return max_len

# Q329 Longest Increasing Path in a Matrix
# Time Limit Exceeded
class Solution1:
    def longestIncreasingPath(self, matrix):

        def check(rowdx,coldy,row,col):
            if 0 <= rowdx < len(matrix) and 0 <= coldy < len(matrix[0]) and matrix[rowdx][coldy] > matrix[row][col]:
                return True
            return False

        def dfs(row,col):
            if not any(check(row+dx,col+dy,row,col) for dx,dy in directions):
                return 1
            # visited.add((row,col))
            return 1 + max(dfs(row+dx,col+dy) for dx,dy in directions if check(row+dx,col+dy,row,col))

        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        maxLength = 0
        for row,seq in enumerate(matrix):
            for col,e in enumerate(seq):
                # visited = set()
                maxLength = max(maxLength,dfs(row,col))
        return maxLength

# Uses dfs and memoization
class Solution2:
    def longestIncreasingPath(self, matrix):
        def dfs(i, j):
            if not dp[i][j]:
                val = matrix[i][j]
                #for each cell in dp, you put the longest path that can be achieved from there; prevents recalculation.
                dp[i][j] = 1 + max(
                    dfs(i - 1, j) if i and val > matrix[i - 1][j] else 0,
                    dfs(i + 1, j) if i < M - 1 and val > matrix[i + 1][j] else 0,
                    dfs(i, j - 1) if j and val > matrix[i][j - 1] else 0,
                    dfs(i, j + 1) if j < N - 1 and val > matrix[i][j + 1] else 0)
            return dp[i][j]

        if not matrix or not matrix[0]:
            return 0
        M, N = len(matrix), len(matrix[0])
        #memoization included
        dp = [[0] * N for i in range(M)]
        return max(dfs(x, y) for x in range(M) for y in range(N))

# Q340 Longest Substring with At Most K Distinct Characters
from collections import defaultdict

class Solution:
    def lengthOfLongestSubstringKDistinct(self, s, k):
        if not s or not k:
            return 0

        used_dict = defaultdict(int)
        length = float('-inf')
        i = j = 0

        while i < len(s):
            if s[i] not in used_dict:
                if len(used_dict) < k:
                    used_dict[s[i]] += 1
                else:
                    while len(used_dict) == k:
                        used_dict[s[j]] -= 1
                        if not used_dict[s[j]]:
                            del used_dict[s[j]]
                            i -= 1
                        j += 1
            else:
                used_dict[s[i]] += 1
            i += 1
            length = max(length,i-j)

        return length

# Q347 Top K Frequent Elements
from collections import Counter

class Solution:
    def topKFrequent(self, nums, k):
        return [k for k, v in Counter(nums).most_common(k)]

# Q348 Design Tic-Tac-Toe
class TicTacToe:

    def __init__(self, n):
        self.row = [0]*n
        self.col = [0]*n
        self.diag = 0
        self.undiag = 0
        self.n = n


    def move(self, row, col, player):
        order = 1 if player == 1 else -1

        self.row[row] += order
        self.col[col] += order

        if row == col:
            self.diag += order

        if col == (self.n-1-row):
            self.undiag += order

        if abs(self.row[row]) == self.n or abs(self.col[col]) == self.n or abs(self.diag) == self.n or abs(self.undiag) == self.n:
            return player
        else:
            return 0
