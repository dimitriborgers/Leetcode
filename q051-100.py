# Q51 N-Queens
class Solution:

    def solveNQueens(self,n):
        board = [[0 for _ in range(n)] for _ in range(n)]

        def solveNQUtil(self, board, col=0):
        # base case: If all queens are placed
            if col >= n:
                return True

            for i in range(n):
                if self.isSafe(board, i, col):
                    board[i][col] = 1

                    if self.solveNQUtil(board, col+1):
                        return True

                    board[i][col] = 0
            return False

        return solveNQUtil(board)

    def isSafe(self, board, row, col):
        # Check this row on left side
        for i in range(col):
            if board[row][i] == 1:
                return False

        # Check upper diagonal on left side
        for i,j in zip(range(row,-1,-1), range(col,-1,-1)):
            if board[i][j] == 1:
                return False

        # Check lower diagonal on left side
        for i,j in zip(range(row,n,1), range(col,-1,-1)):
            if board[i][j] == 1:
                return False

        return True

# Q52 N-Queens II
#Same as Q51

# Q53 Maximum Subarray
class Solution(object):
    def maxSubArray(self, nums):
        if max(nums) < 0:
            return max(nums)
        global_max, local_max = 0, 0
        for x in nums:
            local_max = max(0, local_max + x)
            global_max = max(global_max, local_max)
        return global_max

# Q54 Spiral Matrix
class Solution:
    def spiralOrder(self, matrix):
        result = []
        if matrix == []:
            return result

        left, right, top, bottom = 0, len(matrix[0]) - 1, 0, len(matrix) - 1

        while left <= right and top <= bottom:
            for j in range(left, right + 1):
                result.append(matrix[top][j])
            for i in range(top + 1, bottom):
                result.append(matrix[i][right])
            #instead of right+1,left,-1
            for j in reversed(range(left, right + 1)):
                if top < bottom:
                    result.append(matrix[bottom][j])
            for i in reversed(range(top + 1, bottom)):
                if left < right:
                    result.append(matrix[i][left])
            left, right, top, bottom = left + 1, right - 1, top + 1, bottom - 1
        return result

# Q55 Jump Game
class Solution:
    #Whenever you do recursion, make sure at what level you return
    def canJump(self, nums):
        if len(nums) == 1:
            return True
        for i in range(1,nums[0]+1):
            if self.canJump(nums[0+i:]):
                return True
        return False

# Q56 Merge Intervals
class Solution:
    def merge(self, intervals):
        #intervals = intervals.sort(key=lambda x: x[0])
        intervals = sorted(intervals)
        merged = []
        for i in intervals:
            if not merged or merged[-1][1] < i[0]:
                merged.append(i)
            else:
                merged[-1] = [min(min(merged[-1]),min(i)),max(max(i),max(merged[-1]))]
        return merged

# Q57 Insert Interval
class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

    def __repr__(self):
        return "[{}, {}]".format(self.start, self.end)


class Solution(object):
    def insert(self, intervals, newInterval):
        result = []
        i = 0
        while i < len(intervals) and newInterval.start > intervals[i].end:
            result += intervals[i],
            i += 1
        while i < len(intervals) and newInterval.end >= intervals[i].start:
            newInterval = Interval(min(newInterval.start, intervals[i].start),max(newInterval.end, intervals[i].end))
            i += 1
        result += newInterval,
        result += intervals[i:]
        return result

# Q58 Length of Last Word
class Solution:
    def lengthOfLastWord(self, s):
        return len(s.strip().split(" ")[-1])

# Q59 Spiral Matrix II
class Solution:
    def generateMatrix(self, n):
        total = n**2
        start = 1
        matrix = [[0]*n for _ in range(n)]

        left, right, top, bottom = 0, len(matrix[0]) - 1, 0, len(matrix) - 1

        while left <= right and top <= bottom:
            for j in range(left, right + 1):
                matrix[top][j] = start
                start += 1
            for i in range(top + 1, bottom):
                matrix[i][right] = start
                start += 1
            for j in reversed(range(left, right + 1)):
                if top < bottom:
                    matrix[bottom][j] = start
                    start += 1
            for i in reversed(range(top + 1, bottom)):
                if left < right:
                    matrix[i][left] = start
                    start += 1
            left, right, top, bottom = left + 1, right - 1, top + 1, bottom - 1
        return matrix

# Q60 Permutation Sequence
import math

class Solution:
    def getPermutation(self, n, k):
        seq, k, fact = "", k - 1, math.factorial(n - 1)
        perm = [i for i in range(1, n + 1)]
        for i in reversed(range(n)):
            curr = perm[k // fact]
            seq += str(curr)
            perm.remove(curr)
            if i > 0:
                k %= fact
                fact //= i
        return seq

# Q61
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __repr__(self):
        #Use repr as a recursive function
        return "{} -> {}".format(self.val,repr(self.next))

class Solution:
    def rotateRight(self, head, k):
        if not head or not head.next:
            return head

        for _ in range(k):
            cur = head
            while cur.next.next:
                cur = cur.next
            tail = cur.next
            tail.next = head
            head = tail
            tail = cur
            tail.next = None
        return head

head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = ListNode(5)

# Q62
class Solution:
    def uniquePaths(self, m, n):
        matrix = [[0]*m for _ in range(n)]
        matrix[0][0] = 1
        for i in range(m):
            matrix[0][i] = 1
        for j in range(n):
            matrix[j][0] = 1

        for i in range(1,n):
            for j in range(1,m):
                matrix[i][j] = matrix[i][j-1] + matrix[i-1][j]
        return matrix

# Q63
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid):
        n = len(obstacleGrid)
        m = len(obstacleGrid[0])
        matrix = [[0]*m for _ in range(n)]

        if obstacleGrid[0][0] == 1:
            return 0
        matrix[0][0] = 1

        for i in range(1,m):
            if obstacleGrid[0][i] == 1:
                for j in range(i,m):
                    matrix[0][j] = 0
                break
            else:
                matrix[0][i] = 1

        for i in range(1,n):
            if obstacleGrid[i][0] == 1:
                for j in range(i,n):
                    matrix[j][0] = 0
                break
            else:
                matrix[i][0] = 1

        for i in range(1,n):
            for j in range(1,m):
                if obstacleGrid[i][j] == 1:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = matrix[i][j-1] + matrix[i-1][j]
        return matrix

# Q64
import copy

class Solution:
    def minPathSum(self, grid):
        #global variables don't need to be passed into other functions
        global n
        n = len(grid)
        global m
        m = len(grid[0])
        #list(grid) does not work in this situation because grid is 2d array, and list(grid) would only make a new reference to the outer array, but the inner arrays would still be linked. deepcopy replaces all links.
        result = self._minPathFinder(copy.deepcopy(grid),0,0,0)
        return result

    def _minPathFinder(self,grid,x,y,val):
        grid[x][y] += val

        if x == n-1 and y == m-1:
            return grid[x][y]

        if x < n-1 and y < m-1:
            down = self._minPathFinder(copy.deepcopy(grid),x+1,y,grid[x][y])
            right = self._minPathFinder(copy.deepcopy(grid),x,y+1,grid[x][y])
            return min(down,right)
        elif x < n-1:
            down = self._minPathFinder(copy.deepcopy(grid),x+1,y,grid[x][y])
            return down
        elif y < m-1:
            right = self._minPathFinder(copy.deepcopy(grid),x,y+1,grid[x][y])
            return right

# Q65
import re

class Solution:
    def isNumber(self, s):
        #$ is put at the end of thing you want to match
        return bool(re.match('^\s*[\+-]?((\d+(\.\d*)?)|\.\d+)([eE][\+-]?\d+)?\s*$', s))

# Q66
class Solution:
    def plusOne(self, digits):
        n = len(digits)-1
        digits[n] += 1
        while digits[n] > 9:
            digits[n] = 0
            digits[n-1] += 1
            n = n-1
        return digits

# Q67


# Q68


# Q69


# Q70
