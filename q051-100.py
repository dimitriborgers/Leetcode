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
        return matrix

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
        return matrix

# Q64 Minimum Path Sum
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

# Q65 Valid Number
import re

class Solution:
    def isNumber(self, s):
        #$ is put at the end of thing you want to match
        return bool(re.match('^\s*[\+-]?((\d+(\.\d*)?)|\.\d+)([eE][\+-]?\d+)?\s*$', s))

# Q66 Plus One
class Solution:
    def plusOne(self, digits):
        n = len(digits)-1
        digits[n] += 1
        while digits[n] > 9:
            digits[n] = 0
            digits[n-1] += 1
            n = n-1
        return digits

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
        count = 0

        word = []
        directions = []
        for i in path[::-1]:
            if i == '.':
                directions.append(i)
                if len(directions) > 1:
                    count += 1
                    directions = []
            elif  i == '/' and directions:
                directions.pop()
            elif i.isalpha():
                word.append(i)
            elif word and i == '/':
                if count:
                    word = []
                    count -= 1
                else:
                    word.append(i)
                    return ''.join(word[::-1])
        return '/'

    def simplifyPath2(self, path):
        stack, tokens = [], path.split("/")
        for token in tokens:
            if token == ".." and stack:
                stack.pop()
            elif token != ".." and token != "." and token:
                stack.append(token)
        return "/" + "/".join(stack)

# Q72 Edit Distance
class Solution:
    def minDistance(self, str1, str2,m,n):
        #algorithm good for finding similar words in dictionary
        if m==0:
             return n
        if n==0:
            return m

        if str1[m-1]==str2[n-1]:
            return self.minDistance(str1,str2,m-1,n-1)

        return 1 + min(self.minDistance(str1, str2, m, n-1),    # Insert
                       self.minDistance(str1, str2, m-1, n),    # Remove
                       self.minDistance(str1, str2, m-1, n-1))    # Replace

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
class Solution:
    #infinity
    #minimum = float("inf")
    def minWindow(self, s, t):
        i = 0
        globalWindow = '0'*len(s)
        while i < len(s)-len(t):
            if s[i] not in t:
                i += 1
            else:
                seq = set(t)
                seq.remove(s[i])
                j = i+1
                while seq and j < len(s):
                    if s[j] in seq:
                        seq.remove(s[j])
                    j += 1
                if not seq:
                    localWindow = s[i:j]
                else:
                    localWindow = '0'*len(s)
                i += 1
            globalWindow = localWindow if len(localWindow) < len(globalWindow) else globalWindow
        return globalWindow if globalWindow != '0'*len(s) else ''

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
        n = len(board)
        m = len(board[0])
        locations = []
        for i in range(n):
            for j in range(m):
                if board[i][j] == word[0]:
                    locations.append((i,j))
        for k,v in locations:
            outcome = self._finder(board,word,0,k,v,m,n)
            if outcome:
                return True
        return False

    def _finder(self,board,word,letter,k,v,m,n):
        if letter == len(word):
            return True

        if k < 0 or k >= n or v < 0 or v >= m or board[k][v] != word[letter]:
            return False

        result = self._finder(board,word,letter+1,k+1,v,m,n) or \
                self._finder(board,word,letter+1,k-1,v,m,n) or \
                self._finder(board,word,letter+1,k,v+1,m,n) or \
                self._finder(board,word,letter+1,k,v-1,m,n)
        return result

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
        for i in matrix:
            for j in range(1,len(i)):
                i[0] = int(i[0])
                if int(i[j-1]) > 0 and int(i[j]) > 0:
                    i[j] = int(i[j]) + int(i[j-1])
                else:
                    i[j] = int(i[j])

        maxArea = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                maxArea = matrix[i][j] if matrix[i][j] > maxArea else maxArea
                if i == 0:
                    localArea,localAreaList,temp = 0,[],i
                    while temp < len(matrix):
                        if matrix[temp][j]:
                            localAreaList.append(matrix[temp][j])
                        temp += 1
                    minimum = min(localAreaList)
                    #below loop cannot use i because it would interfere with if i == 0 statement
                    #i would have last value of element in localAreaList
                    for x in localAreaList:
                        if x:
                            localArea += minimum
                    maxArea = localArea if localArea > maxArea else maxArea
        return maxArea

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
        result = []
        self.restoreIpAddressesRecur(result, s, 0, "", 0)
        return result

    def restoreIpAddressesRecur(self, result, s, start, current, dots):
        # pruning to improve performance
        if (4 - dots) * 3 < len(s) - start or (4 - dots) > len(s) - start:
            return

        if start == len(s) and dots == 4:
            #if you just write append(current), there is an extra dot at the end
            result.append(current[:-1])
        else:
            for i in xrange(start, start + 3):
                if len(s) > i and self.isValid(s[start:i + 1]):
                    current += s[start:i + 1] + '.'
                    self.restoreIpAddressesRecur(result, s, i + 1, current, dots + 1)
                    #same thing as creating a temp of current before any changes are made and ressasgning current to it
                    current = current[:-(i - start + 2)]

    def isValid(self, s):
        if len(s) == 0 or (s[0] == '0' and s != "0"):
            return False
        return int(s) < 256

# Q94 Binary Tree Inorder Traversal
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def inorderTraversalRec(self,root):
        global result
        result = []
        return self._inorder_helper(root)

    def _inorder_helper(self, root):
        if root.left:
            self._inorder_helper(root.left)
        #can't be append(root) because this would traverse from root every time when you print
        result.append(root.val)
        if root.right:
            self._inorder_helper(root.right)
        return result

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
        pre, cur = None, root

        while cur:
            if cur.left is None:
                self.detectBroken(broken, pre, cur)
                pre = cur
                cur = cur.right
            else:
                node = cur.left
                while node.right and node.right != cur:
                    node = node.right

                if node.right is None:
                    node.right =cur
                    cur = cur.left
                else:
                    self.detectBroken(broken, pre, cur)
                    node.right = None
                    pre = cur
                    cur = cur.right

        broken[0].val, broken[1].val = broken[1].val, broken[0].val

        return root

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