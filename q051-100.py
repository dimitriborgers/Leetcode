# Q51 N-Queens
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

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
class Solution:
    def maxSubArray(self, nums):
        lmax = gmax = nums[0]
        for pos,e in enumerate(nums):
            if pos > 0:
                if e > lmax and lmax < 0:
                    lmax = e
                else:
                    lmax += e
                gmax = max(gmax,lmax)

        return gmax

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
#Time: O(nlogn)
#Space: O(1) if you mutate given array instead of creating merged[]
class Solution:
    def merge(self, intervals):
        #intervals = intervals.sort(key=lambda x: x[0])
        intervals.sort(key=lambda x: x.start)
        merged = []
        for interval in intervals:
            #sequence[-1] great way to get last element in list
            if not merged or merged[-1].end < interval.start:
                merged.append(interval)
            else:
                merged[-1].end = max(merged[-1].end, interval.end)
        return merged

#Using adjacency list
class Solution:
    def overlap(self, a, b):
        return a.start <= b.end and b.start <= a.end

    # generate graph where there is an undirected edge between intervals u
    # and v iff u and v overlap.
    def build_graph(self, intervals):
        graph = collections.defaultdict(list)

        for i, interval_i in enumerate(intervals):
            for j in range(i+1, len(intervals)):
                if self.overlap(interval_i, intervals[j]):
                    graph[interval_i].append(intervals[j])
                    graph[intervals[j]].append(interval_i)

        return graph

    # merges all of the nodes in this connected component into one interval.
    def merge_nodes(self, nodes):
        min_start = min(node.start for node in nodes)
        max_end = max(node.end for node in nodes)
        return Interval(min_start, max_end)

    # gets the connected components of the interval overlap graph.
    def get_components(self, graph, intervals):
        visited = set()
        comp_number = 0
        nodes_in_comp = collections.defaultdict(list)

        def mark_component_dfs(start):
            stack = [start]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    nodes_in_comp[comp_number].append(node)
                    stack.extend(graph[node])

        # mark all nodes in the same connected component with the same integer.
        for interval in intervals:
            if interval not in visited:
                mark_component_dfs(interval)
                comp_number += 1

        return nodes_in_comp, comp_number

    def merge(self, intervals):
        graph = self.build_graph(intervals)
        nodes_in_comp, number_of_comps = self.get_components(graph, intervals)

        # all intervals in each connected component must be merged.
        return [self.merge_nodes(nodes_in_comp[comp]) for comp in range(number_of_comps)]

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
        #while statements can have else flows as well
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

# Q61 Rotate List
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

# Q62 Unique Paths
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
        return matrix[-1][-1]

# Q63 Unique Paths II
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
        return matrix[-1][-1]

# Q64 Minimum Path Sum
class Solution:
    def minPathSum(self, grid):
        for pos,e in enumerate(grid[0]):
            if pos:
                grid[0][pos] += grid[0][pos-1]
        for pos,e in enumerate(grid):
            if pos:
                grid[pos][0] += grid[pos-1][0]

        for i in range(1,len(grid)):
            for j in range(1,len(grid[0])):
                grid[i][j] += min(grid[i-1][j],grid[i][j-1])

        return grid[-1][-1]

# Q65 Valid Number
import re

class Solution:
    def isNumber(self, s):
        #$ is put at the end of thing you want to match
        return bool(re.match('^\s*[\+-]?((\d+(\.\d*)?)|\.\d+)([eE][\+-]?\d+)?\s*$', s))

# Q66 Plus One
class Solution:
    def plusOne(self, digits):
        total = 0
        for pos,e in enumerate(digits):
            total = total*10 + e
        total += 1
        result = []
        while total:
            tmp = total % 10
            result.append(tmp)
            total //= 10
        return result[::-1]

# Q67 Add Binary
import itertools

class Solution:
    def addBinary(self, a, b):
        a = reversed(a)
        b = reversed(b)
        c = []
        overflow = None

        for x,y in itertools.zip_longest(a,b,fillvalue=0):
            if overflow:
                quotient,remainder = divmod(int(x)+int(y)+overflow,2)
                if not remainder:
                    overflow = 1
                else:
                    overflow = 0
                c.append(str(remainder))
            else:
                quotient,remainder = divmod(int(x)+int(y),2)
                if not remainder:
                    overflow = 1
                else:
                    overflow = 0
                c.append(str(remainder))
        if overflow:
            c.append('1')
        #or return ''.join(c[::-1])
        return ''.join(reversed(c))

# Q68 Text Justification
class Solution:
    def fullJustify(self, words, maxWidth):

        def addSpaces(i, spaceCnt, maxWidth, is_last):
            if i < spaceCnt:
                return 1 if is_last else (maxWidth // spaceCnt) + int(i < maxWidth % spaceCnt)
            return 0

        def connect(words, maxWidth, begin, end, length, is_last):
            s = []
            n = end - begin
            for i in range(n):
                s += words[begin + i],
                s += ' ' * addSpaces(i, n - 1, maxWidth - length, is_last),
            line = "".join(s)
            if len(line) < maxWidth:
                line += ' ' * (maxWidth - len(line))
            return line

        res = []
        begin, length = 0, 0
        for i in range(len(words)):
            if length + len(words[i]) + (i - begin) > maxWidth:
                res += connect(words, maxWidth, begin, i, length, False),
                begin, length = i, 0
            length += len(words[i])

        res += connect(words, maxWidth, begin, len(words), length, True),
        return res

# Q69 Sqrt(x)
import itertools

class Solution:
    def mySqrt(self, x):

        if x < 2:
            return x

        counter = itertools.count()
        #return next(counter)
        for i in counter:
            if i**2 > x:
                return i-1

# Q70 Climbing Stairs
class Solution:
    def climbStairs(self, n):
        if n == 2:
            return 2
        if n == 1:
            return 1

        return self.climbStairs(n-1)+self.climbStairs(n-2)

    def climbStairs2(self,n):
        if n == 1:
            return 1

        dp = [0]*(n)
        dp[0] = 1
        dp[1] = 2
        for i in range(2,n):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n-1]

# Q71 Simplify Path
class Solution:
    #str class does not have del method
    def simplifyPath(self, path):
        stack, tokens = [], path.split("/")
        for token in tokens:
            if token == ".." and stack:
                stack.pop()
            elif token != ".." and token != "." and token:
                stack.append(token)
        return "/" + "/".join(stack)

# Q72 Edit Distance
class Solution:
    def minDistance(self, word1, word2):
        m = len(word1)
        n = len(word2)
        table = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            table[i][0] = i
        for j in range(n + 1):
            table[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    table[i][j] = table[i - 1][j - 1]
                else:
                    table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1], table[i - 1][j - 1])
        return table[-1][-1]

# Q73 Set Matrix Zeroes
class Solution:
    def setZeroes(self, matrix):
        n = len(matrix)
        m = len(matrix[0])

        new = []

        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    new.append((i,j))

        for x,y in new:
            tempx = x
            tempx1 = x
            tempy = y
            tempy1 = y
            while tempx < n:
                matrix[tempx][y] = 0
                tempx+=1
            while tempx1 >= 0:
                matrix[tempx1][y] = 0
                tempx1-=1
            while tempy < m:
                matrix[x][tempy] = 0
                tempy+=1
            while tempy1 >= 0:
                matrix[x][tempy1] = 0
                tempy1-=1
        return matrix

# Q74 Search a 2D Matrix
class Solution:
    def searchMatrix(self, matrix, target):
        #always handle errors!
        if not matrix:
            return -1

        left,right = 0, len(matrix)-1

        while left <= right:
            mid = left + ((right-left) // 2)
            if matrix[mid][0] == target:
                return matrix[mid][0]
            elif matrix[mid][0] > target:
                right = mid - 1
            else:
                left = mid + 1

        if matrix[mid][0] < target:
            row = matrix[mid+1]
        else:
            row = matrix[mid-1]

        left,right = 0, len(matrix[0])-1

        while left <= right:
            mid2 = left + ((right-left) // 2)
            if row[mid2] == target:
                return row[mid2]
            elif row[mid2] > target:
                right = mid2 - 1
            else:
                left = mid2 + 1

        return -1

# Q75 Sort Colors
class Solution:
    def sortColors(self, nums):
        def triPartition(nums, target):
            i, j, n = 0, 0, len(nums) - 1

            while j <= n:
                if nums[j] < target:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
                    j += 1
                elif nums[j] > target:
                    nums[j], nums[n] = nums[n], nums[j]
                    n -= 1
                else:
                    j += 1

        triPartition(nums, 1)
        return nums

# Q76 Minimum Window Substring
from collections import defaultdict,Counter

class Solution:
    def minWindow(self, s, t):
        if not t or not s:
            return ""

        dict_t = Counter(t)
        required = len(dict_t)
        l, r = 0, 0
        formed = 0
        window_counts = defaultdict(int)
        ans = float("inf"), None, None

        while r < len(s):
            character = s[r]
            window_counts[character] += 1

            if character in dict_t and window_counts[character] == dict_t[character]:
                formed += 1

            while l <= r and formed == required:
                character = s[l]
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)
                window_counts[character] -= 1
                if character in dict_t and window_counts[character] < dict_t[character]:
                    formed -= 1
                l += 1
            r += 1
        return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]

# Q77 Combinations
import itertools

class Solution:
    def combine(self, n, k):
        return itertools.combinations(n,k)

# Q78 Subsets
class Solution:
    def subsets(self, nums):
        #enumerate(nums) can replace zip(range(len(s)),s)
        return [[x for (x,pos) in zip(nums,range(len(nums))) if (2**pos) & a]for a in range(2**len(nums))]

# Q79 Word Search
class Solution:
    def exist(self, board, word):

        def dfs(word,r,c):
            if not word:
                return True

            if 0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == word[0]:

                #You don't want to add elements to a visited set because you're searching for a specific pattern, while a regular dfs just finds connected components in any order.
                #Thus, if searching a specific pattern, you have to put the element back into the grid after every call.
                tmp = board[r][c]
                board[r][c] = '#'
                #doing or statements is faster than any(list of calls)
                result = dfs(word[1:],r+1,c) or dfs(word[1:],r-1,c) or dfs(word[1:],r,c+1) or dfs(word[1:],r,c-1)
                board[r][c] = tmp
                return result

        for r,row in enumerate(board):
            for c,col in enumerate(row):
                if col == word[0]:
                    #instead of doing:
                    #res = dfs()
                    #if res:
                        #return True
                    if dfs(word,r,c):
                        return True
        return False

# Q80 Remove Duplicates from Sorted Array II
class Solution:
    def removeDuplicates(self, nums):
        for i in range(len(nums)):
            while i < len(nums)-2 and nums[i+2] == nums[i]:
                del nums[i+2]
        return nums

# Q81 Search in Rotated Sorted Array II
class Solution:
    def search(self, nums, target):
        if not nums:
            return False

        left, right = 0,len(nums)-1

        while left <= right:
            mid = left + (right-left)//2

            if nums[mid] == target:
                return True
            elif nums[mid] < target and nums[right] < target:
                right = mid - 1
            else:
                left = mid + 1
        return False

# Q82 Remove Duplicates from Sorted List II
class Solution:
    def deleteDuplicates(self,head):
        #Doesn't matter if below variables are before or after _delete_helper definition
        previous = ListNode(-1)
        previous.next = head
        def _delete_helper(head,previous):
            duplicate = False
            cur = head
            while cur.next:
                if cur.next.val == cur.val:
                    cur = cur.next
                    duplicate = True
                else:
                    if duplicate:
                        cur = cur.next
                        #internal methods don't need self in front of function use
                        cur.next = _delete_helper(cur.next,previous)
                        return cur
                    else:
                        head.next = _delete_helper(cur.next,head)
                        return head
            if duplicate:
                return None
            else:
                return cur
        return _delete_helper(head,previous)

# Q83 Remove Duplicates from Sorted List
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __repr__(self):
        return '{} -> {}'.format(self.val,repr(self.next))

class Solution:
    def deleteDuplicates(self, head):
        cur = head
        while cur.next:
            #if you did cur.next = something, head.next would also change.
            #This doesn't happen here because just changing cur location.
            if cur.next.val == cur.val:
                cur = cur.next
            else:
                head.next = self.deleteDuplicates(cur.next)
                return head
        return cur

# Q84 Largest Rectangle in Histogram
class Solution:
    def largestRectangleArea(self, heights):
        i,j = 0,len(heights)-1
        maxArea = 0

        while i <= j:
            left = heights[i]
            right = heights[j]
            width = j - i + 1
            height = min(heights[i:j+1])
            maxArea = width * height if width * height > maxArea else maxArea
            if left < right:
                i += 1
            else:
                j -= 1
        return maxArea

# Q85 Maximal Rectangle
class Solution:
    def maximalRectangle(self, matrix):
        if not matrix or not matrix[0]:
            return 0
        m, n = len(matrix), len(matrix[0])
        dp = [[0]*n for _ in range(m)]
        res = 0
        for j in range(n):
            dp[0][j] = 1 if matrix[0][j]=="1" else 0
        for i in range(0, m):
            for j in range(n):
                if i>0:
                    dp[i][j] = dp[i-1][j] + 1 if matrix[i][j]=="1" else 0
                local_min = dp[i][j]
                res = max(res, dp[i][j])
                for s in range(j, -1, -1):
                    if dp[i][s]==0:
                        break
                    local_min = min(local_min, dp[i][s])
                    res = max(res, local_min*(j-s+1))
        return res

class Solution1:
    def maximalRectangle(self, matrix):
        for r,row in enumerate(matrix):
            for c,elem in enumerate(row):
                matrix[r][c] = int(matrix[r][c])
                if c > 0 and matrix[r][c]:
                    matrix[r][c] += matrix[r][c-1]

        secondary = [row[:] for row in matrix]

        for r,row in enumerate(matrix):
            for c,elem in enumerate(row):
                if r > 0 and matrix[r][c] and matrix[r-1][c]:
                    matrix[r][c] = min(secondary[r][c],secondary[r-1][c])+matrix[r-1][c]

        return max(max(row) for row in matrix)

# Q86 Partition List
class Solution:
    def partition(self, head, x):
        dummy = ListNode(-1)
        dummy.next = head
        cur = dummy
        begin = dummy
        end = None
        previous = None
        while cur.next:
            if cur.next.val >= x and not end:
                end = cur.next
                cur = cur.next
            elif cur.next.val < x and end:
                previous = cur
                temp = cur.next.next
                dummy.next = cur.next
                dummy.next.next = end
                cur.next = temp
                dummy = dummy.next
            elif cur.next.val < x and not end:
                dummy = dummy.next
                cur = cur.next
            else:
                cur = cur.next
        if cur.val < x:
            dummy.next = cur
            cur.next = end
            previous.next = None
        return begin.next

# Q87 Scramble String
class Solution:
    def isScramble(self, s1, s2):
        def recur(word):
            if len(word) == 1:
                return word
            if len(word) == 2:
                #str does not support item assignment
                word = list(word)
                word[0],word[1] = word[1],word[0]
                return ''.join(word)
            mid = len(word) // 2
            left = word[:mid]
            right = word[mid:]
            return recur(left) + recur(right)
        result = list(recur(s1))
        outcome = []
        for i in range(len(result)-1):
            temp = list(result)
            j = i+2
            k = i-1
            while j < len(result):
                temp[j] = s1[j]
                j += 1
            while k >= 0:
                temp[k] = s1[k]
                k -= 1
            outcome.append(''.join(temp))
        return True if s2 in outcome else False

# Q88 Merge Sorted Array
class Solution:
    def merge(self, nums1, m, nums2, n):
        i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] <= nums2[j] and nums1[i] != 0:
                i+=1
            else:
                #list.insert(pos, elmnt)
                nums1.insert(i,nums2[j])
                del nums1[-1]
                j+=1
        return nums1

# Q89 Gray Code
class Solution:
    def grayCode(self, n):
        global result
        result,binary = [],['0']*n
        #int('111', 2) converts binary to decimal,bin(255) -> '0b11111111'
        result.append(int(''.join(binary),2))
        return self._recur(binary,0)

    def _recur(self,binary,position):
        for i in range(position,len(binary)):
            binary[i] = '1'
            result.append(int(''.join(binary),2))
            self._recur(list(binary),i+1)
            binary[i] = '0'
        return result

class Solution2:
    def grayCode(self, n):
        result = [0]
        for i in range(n):
            #has to be reversed because as list gets appended, you don't start over for loop until reach end of loop that you started
            #in second loop, reversed result starts [1,0], but even though 3 is appended, you still complete the loop and reach 0, even if reversed result now looks like [3,1,0].
            #if you didn't reverse, loop would never end due to appending
            for x in reversed(result):
                print(result)
                print(x)
                #1 << 1 | 1 == 3 because you do left to right
                result.append(1 << i | x)
        return result

# Q90 Subsets II
class Solution:
    def subsetsWithDup(self, nums):
        outcome = [[x for (x,pos) in zip(nums,range(len(nums))) if 2**pos & b]for b in range(2**len(nums))]
        #you can create a set of frozensets. The only difference between a set and a frozenset is that the latter is immutable
        result = set()
        for i in outcome:
            temp = []
            for j in i:
                temp.append(j)
            #to create a set of lists, you have to change a list into tuples and then add
            temp = tuple(temp)
            result.add(temp)
        return result

# Q91 Decode Ways
class Solution:
    def numDecodings(self, s):
        #very similar to fibonacci
        if len(s) == 0 or s[0] == '0':
            return 0
        prev, prev_prev = 1, 0
        for i in range(len(s)):
            cur = 0
            if s[i] != '0':
                cur = prev
            #Create an and statement that has two ors in it
            if i > 0 and (s[i-1] == '1' or (s[i-1] == '2' and s[i] <= '6')):
                cur += prev_prev
            prev, prev_prev = cur, prev
        return prev

# Q92 Reverse Linked List II
class Solution:
    def reverseBetween(self, head, m, n):
        if m == n or not head.next:
            return head
        if m > n:
            m,n = n,m

        dummy = ListNode(-1)
        dummy.next = head
        dummy_global = dummy
        i = j = 1
        curi = head

        while curi.next and i < m:
            curi = curi.next
            dummy = dummy.next
            i += 1
            j += 1
        curj = curi
        while curj.next and j < n:
            curj = curj.next
            j += 1

        while i < j:
            old_next = curi.next
            dummy_next = dummy.next
            dummy.next = old_next
            curi.next = old_next.next
            old_next.next = dummy_next
            head = dummy_global.next
            i += 1

        return head

# Q93 Restore IP Addresses
class Solution:
    def restoreIpAddresses(self, s):
        res = []
        self.dfs(s, 0, "", res)
        return res

    def dfs(self, s, index, path, res):
        if index == 4:
            if not s:
                res.append(path[:-1])
            return

        for i in range(1, 4):
            if i <= len(s):
                if i == 1:
                    self.dfs(s[i:], index+1, path+s[:i]+".", res)
                elif i == 2 and s[0] != "0":
                    self.dfs(s[i:], index+1, path+s[:i]+".", res)
                elif i == 3 and s[0] != "0" and int(s[:3]) <= 255:
                    self.dfs(s[i:], index+1, path+s[:i]+".", res)

# Q94 Binary Tree Inorder Traversal
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def inorderTraversal(self,root):
        self.result = []
        return self._inorder_helper(root)

    def _inorder_helper(self, root):
        if root and root.left:
            self._inorder_helper(root.left)
        if root:
            self.result.append(root.val)
        if root and root.right:
            self._inorder_helper(root.right)
        return self.result

    def inorderTraversalIterative(self, root):
        result, curr = [], root
        while curr:
            if not curr.left:
                result.append(curr.val)
                curr = curr.right
            else:
                node = curr.left
                while node.right and node.right != curr:
                    node = node.right

                if not node.right:
                    node.right = curr
                    curr = curr.left
                else:
                    result.append(curr.val)
                    node.right = None
                    curr = curr.right
        return result

# Q95 Unique Binary Search Trees II
class Solution:
    def numTrees(self, n):

        G = [0]*(n+1)
        G[0], G[1] = 1, 1

        for i in range(2, n+1):
            for j in range(1, i+1):
                G[i] += G[j-1] * G[i-j]
        return G[n]

# Q96 Unique Binary Search Trees
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def __repr__(self):
        if self:
            serial = []
            queue = [self]

            while queue:
                cur = queue[0]

                if cur:
                    serial.append(cur.val)
                    queue.append(cur.left)
                    queue.append(cur.right)
                else:
                    serial.append("#")

                queue = queue[1:]

            while serial[-1] == "#":
                serial.pop()

            return repr(serial)

        else:
            return None

class Solution:
    def generateTrees(self, n):
        return self.generateTreesRecu(1, n)

    # Doing this with my normal recursion becomes to complicated
    def generateTreesRecu(self, low, high):
        result = []
        if low > high:
            result.append(None)
        for i in range(low, high + 1):
            left = self.generateTreesRecu(low, i - 1)
            right = self.generateTreesRecu(i + 1, high)
            for j in left:
                for k in right:
                    cur = TreeNode(i)
                    cur.left = j
                    cur.right = k
                    result.append(cur)
        return result

# Q97 Interleaving String
class Solution:
    def isInterleave(self, s1, s2, s3):
        self.match = {}
        if len(s1) + len(s2) != len(s3):
            return False
        return self.isInterleaveRecu(s1, s2, s3, 0, 0, 0)

    def isInterleaveRecu(self, s1, s2, s3, a, b, c):
        if repr([a, b]) in self.match.keys():
            return self.match[repr([a, b])]

        if c == len(s3):
            return True

        result = False
        if a < len(s1) and s1[a] == s3[c]:
            result = result or self.isInterleaveRecu(s1, s2, s3, a + 1, b, c + 1)
        if b < len(s2) and s2[b] == s3[c]:
            result = result or self.isInterleaveRecu(s1, s2, s3, a, b + 1, c + 1)

        self.match[repr([a, b])] = result

        return result

# Q98 Validate Binary Search Tree
class Solution:
    def isValidBST(self, root):
        if not root:
            return True

        if root.left and root.left.val > root.val:
            return False
        if root.right and root.right.val < root.val:
            return False
        return self.isValidBST(root.right) and self.isValidBST(root.left)

# Q99 Recover Binary Search Tree
class Solution:
    def recoverTree(self, root):
        return self.MorrisTraversal(root)

    def MorrisTraversal(self, root):
        if root is None:
            return

        broken = [None, None]
        # pre is the predecessor of current in the type of traversal you are doing (inorder, preorder, postorder)
        # Since this problem is with BST, we use inorder
        pre, cur = None, root

        while cur:
            if not cur.left:
                self.detectBroken(broken, pre, cur)
                pre = cur
                cur = cur.right
            else:
                node = cur.left
                while node.right and node.right != cur:
                    node = node.right

                if not node.right:
                    node.right = cur
                    cur = cur.left
                else:
                    self.detectBroken(broken, pre, cur)
                    node.right = None
                    pre = cur
                    cur = cur.right

        broken[0].val, broken[1].val = broken[1].val, broken[0].val

    def detectBroken(self, broken, pre, cur):
        if pre and pre.val > cur.val:
            if broken[0] is None:
                broken[0] = pre
            broken[1] = cur

# Q100 Same Tree
class Solution:
    def isSameTree(self, p, q):
        if p is None and q is None:
            return True

        if p is not None and q is not None:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

        return False
