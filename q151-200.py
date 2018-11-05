# 151 Reverse Words in a String
class Solution:
    def reverseWords(self, s):
        return ' '.join(s.split(' ')[::-1])

# 152 Maximum Product Subarray
class Solution:
    #local and global max/min is often useful
    def maxProduct(self, A):
        global_max, local_max, local_min = float("-inf"), 1, 1
        for x in A:
            local_max = max(1, local_max)
            if x > 0:
                local_max, local_min = local_max * x, local_min * x
            else:
                #local_min is only different if we encounter a negative value
                local_max, local_min = local_min * x, local_max * x
            global_max = max(global_max, local_max)
        return global_max

# 153 Find Minimum in Rotated Sorted Array
class Solution:
    def findMin(self, nums):
        minimum,left,right = nums[0],0,len(nums)-1
        while left != right:
            mid = left + (right-left) // 2
            if nums[right] < nums[mid]:
                minimum = nums[mid] if nums[mid] < minimum else minimum
                left = mid + 1
            else:
                minimum = nums[mid] if nums[mid] < minimum else minimum
                right = mid
        if nums[right] < minimum:
            minimum = nums[right]
        return minimum

# 154 Find Minimum in Rotated Sorted Array II
class Solution:
    #same as without duplicates
    def findMin(self, nums):
        minimum,left,right = nums[0],0,len(nums)-1

        while left < right:
            mid = left + (right-left) // 2
            if nums[right] < nums[mid]:
                minimum = nums[mid] if nums[mid] < minimum else minimum
                left = mid + 1
            else:
                minimum = nums[mid] if nums[mid] < minimum else minimum
                right = mid
        if nums[right] < minimum:
            minimum = nums[right]
        return minimum

# 155 Min Stack
class MinStack:

    def __init__(self):
        self.stack = []
        self.size = 0
        self.min = float('inf')

    def __repr__(self):
        return '{}'.format(self.stack)

    def push(self, x):
        self.stack.append(x)
        self.min = x if x < self.min else self.min

    def pop(self):
        return self.stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        if not self.stack:
            return 'Empty stack'
        return self.min

# 156 Binary Tree Upside Down
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
    def upsideDownBinaryTree(self, root):
        dummy = TreeNode(-1)
        dummy.left = root
        prev,cur = root,root.left
        while cur:
            cur_left = cur.left
            cur_right = cur.right
            cur.left = prev.right
            cur.right = dummy.left
            dummy.left = cur
            prev.left = cur_left
            prev.right = cur_right
            root = dummy.left
            prev,cur = prev,prev.left
        return root

# 157 Read N Characters Given Read4
def read4(buf):
    global file_content
    i = 0
    while i < len(file_content) and i < 4:
        buf[i] = file_content[i]
        i += 1

    if len(file_content) > 4:
        file_content = file_content[4:]
    else:
        file_content = ""
    return i

class Solution:
    def read(self, buf, n):
        read_bytes = 0
        buffer = [''] * 4
        for i in range(n // 4 + 1):
            size = read4(buffer)
            if size:
                size = min(size, n-read_bytes)
                buf[read_bytes:read_bytes+size] = buffer[:size]
                read_bytes += size
            else:
                break
        return read_bytes

if __name__ == "__main__":
    global file_content
    buf = ['' for _ in range(100)]
    file_content = "a"
    print(buf[:Solution().read(buf, 9)])
    file_content = "abcdefghijklmnop"
    print(buf[:Solution().read(buf, 9)])

# 158 Read N Characters Given Read4 II - Call multiple times
def read4(buf):
    global file_content
    i = 0
    while i < len(file_content) and i < 4:
        buf[i] = file_content[i]
        i += 1

    if len(file_content) > 4:
        file_content = file_content[4:]
    else:
        file_content = ""
    return i

class Solution:
    def __init__(self):
        self.__buf4 = [''] * 4
        self.__i4 = 0
        self.__n4 = 0

    def read(self, buf, n):
        i = 0
        while i < n:
            if self.__i4 < self.__n4:  # Any characters in buf4.
                buf[i] = self.__buf4[self.__i4]
                i += 1
                self.__i4 += 1
            else:
                #new 4 letters must be requested
                self.__n4 = read4(self.__buf4)  # Read more characters.
                if self.__n4:
                    self.__i4 = 0
                else:  # Buffer has been empty.
                    break

        return i

if __name__ == "__main__":
    global file_content
    file_content = "ab"
    sol = Solution()
    buf = ['' for _ in range(100)]
    print(buf[:sol.read(buf, 1)])
    print(buf[:sol.read(buf, 2)])

# 159 Longest Substring with At Most Two Distinct Characters
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s):
        output = float('-inf')
        sets, lists = set(), []
        i = j = 0
        while i < len(s):
            if s[i] not in sets and len(sets) < 2:
                sets.add(s[i])
                lists.append(s[i])
            elif s[i] not in sets and len(sets) == 2:
                while len(sets) == 2:
                    lists.remove(s[j])
                    if not lists.count(s[j]):
                        sets.remove(s[j])
                    j += 1
                i -= 1
            else:
                lists.append(s[i])
            output = max(output,len(lists))
            i += 1
        return output

# 160 Intersection of Two Linked Lists
class Solution:
    def getIntersectionNode(self, headA, headB):
        curA, curB = headA, headB
        begin, tailA, tailB = None, None, None

        while curA and curB:
            if curA is curB:
                begin = curA
                break

            if curA.next:
                curA = curA.next
            elif tailA is None:
                tailA = curA
                curA = headB
            else:
                break

            if curB.next:
                curB = curB.next
            elif tailB is None:
                tailB = curB
                curB = headA
            else:
                break

        return begin

# 161 One Edit Distance
class Solution:
    def isOneEditDistance(self, s, t):
        m, n = len(s), len(t)
        if m > n:
            return self.isOneEditDistance(t, s)
        if n - m > 1:
            return False

        i, shift = 0, n - m
        while i < m and s[i] == t[i]:
            i += 1
        if shift == 0:
            i += 1
        while i < m and s[i] == t[i + shift]:
            i += 1

        return i == m

# 162 Find Peak Element
class Solution:
    #Very important to know what left and right become.
    #Sometimes it's right = mid + 1, others right = mid
    def findPeakElement(self, nums):
        left, right = 0, len(nums) - 1

        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        return left

# 163 Missing Ranges
class Solution:
    def findMissingRanges(self, nums, lower, upper):
        output = []
        for i in range(len(nums)-1):
            if nums[i+1] - nums[i] == 2:
                output.append(str(nums[i]+1))
            elif nums[i+1] - nums[i] > 1:
                #can do output.append('{}->{}'.format(nums[i]+1,nums[i+1]-1)
                output.append(str(nums[i]+1)+'->'+str(nums[i+1]-1))
            else:
                continue
        if upper > nums[-1]:
            output.append(str(nums[-1]+1)+'->'+str(upper))
        return output

# 164 Maximum Gap
class Solution:
    #sets() do not keep things in sorted order. You can iterate through set.
    #Radix Sort uses counting sort for each significant digit
    def maximumGap(self, nums):
        if len(nums) < 2:
            return 0
        maximum = float('-inf')
        nums = self.radixSort(nums)
        for i in range(1,len(nums)):
            maximum = nums[i]-nums[i-1] if nums[i]-nums[i-1] > maximum else maximum
        return maximum

    def countingSort(self,arr, exp1):
        output = [0] * (len(arr))
        #only needs to go to 10 since each digit is 0-9
        count = [0] * (10)

        for i in range(len(arr)):
            index = (arr[i] // exp1)
            count[(index) % 10] += 1

        #the max element in count is the length of array we are sorting.
        for i in range(1,10):
            count[i] += count[i-1]

        i = len(arr)-1
        while i>=0:
            index = (arr[i] // exp1)
            output[count[(index) % 10] - 1] = arr[i]
            count[(index) % 10] -= 1
            i -= 1

        for i in range(len(arr)):
            arr[i] = output[i]

    def radixSort(self,arr):
        max1 = max(arr)
        exp = 1
        while max1 // exp > 0:
            self.countingSort(arr,exp)
            exp *= 10
        return arr

# 165 Compare Version Numbers
class Solution:
    def compareVersion(self, version1, version2):
        if len(version1) < len(version2):
            self.compareVersion(version2,version1)

        version1 = version1.split('.')
        version2 = version2.split('.')

        for i in range(len(version2)):
            if version1[i] > version2[i]:
                return 1
            elif version1[i] < version2[i]:
                return -1
            else:
                continue
        if len(version1) > len(version2):
            return 1

# 166 Fraction to Recurring Decimal
class Solution:
    def fractionToDecimal(self, numerator, denominator):
        result = ""
        if (numerator > 0 and denominator < 0) or (numerator < 0 and denominator > 0):
            result = "-"

        dvd, dvs = abs(numerator), abs(denominator)
        result += str(dvd // dvs)
        dvd %= dvs

        if dvd > 0:
            result += "."

        lookup = {}
        while dvd and dvd not in lookup:
            lookup[dvd] = len(result)
            dvd *= 10
            result += str(dvd // dvs)
            dvd %= dvs

        if dvd in lookup:
            result = result[:lookup[dvd]] + "(" + result[lookup[dvd]:] + ")"

        return result

# 167 Two Sum II - Input array is sorted
class Solution:
    def twoSum(self, numbers, target):
        i,j = 0,len(numbers)-1
        while i < j:
            if numbers[i] + numbers[j] > target:
                j -= 1
            elif numbers[i] + numbers[j] < target:
                i += 1
            else:
                return (i,j)
        return -1

# 168 Excel Sheet Column Title
class Solution:
    def convertToTitle(self, n):
        result, dvd = "", n
        while dvd:
            result += chr((dvd - 1) % 26 + ord('A'))
            dvd = (dvd - 1) // 26
        return result[::-1]

    def convertToTitle2(self, n):
        result = ""
        while n // 26:
            result += chr((n // 26) - 1 + ord('A'))
            n %= 26
        if n:
            result += chr(n - 1 + ord('A'))
        return result

# 169 Majority Element
class Solution:
    def majorityElement(self, nums):
        for i in nums:
            if nums.count(i) > len(nums)//2:
                return i

# 170 Two Sum III - Data structure design
class TwoSum:
    def __init__(self):
        self.seq = []
        self.size = 0

    def add(self, number):
        self.seq.append(number)
        self.seq.sort()
        self.size += 1

    def find(self, value):
        if self.size == 0:
            return "No elements stored"

        i,j = 0,len(self.seq)-1
        while i < j:
            if self.seq[i] + self.seq[j] > value:
                j -= 1
            if self.seq[i] + self.seq[j] < value:
                i += 1
            else:
                return True
        return False

# 171 Excel Sheet Column Number
class Solution:
    def titleToNumber(self, s):
        result = 0
        for i in s:
            result += result*25 + (ord(i)+1-ord('A'))
        return result

# 172 Factorial Trailing Zeroes
import math

class Solution:
    def trailingZeroes(self, n):
        n = math.factorial(n)
        result = 0
        while not n % 10:
            result += 1
            n //= 10
        return result

# 173 Binary Search Tree Iterator
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class BSTIterator:
    def __init__(self, root):
        self.cur = root
        self.stack = []

    def hasNext(self):
        if self.cur.left or self.cur.right:
            return True
        return False

    def next(self):
        while self.cur:
            self.stack.append(self.cur)
            self.cur = self.cur.left
        self.cur = self.stack.pop()
        node = self.cur
        self.cur = self.cur.right

        return node.val

# 174 Dungeon Game
import copy

class Solution:
    def calculateMinimumHP(self, dungeon):
        return abs(self.calculateMinimumHPRec(dungeon,0,0,dungeon[0][0]))+1

    def calculateMinimumHPRec(self,dungeon,i,j,minimum):
        print(dungeon,minimum)
        right = bottom = float('-inf')
        prev_min = minimum

        if j < len(dungeon[0])-1:
            dungeon[i][j+1] += dungeon[i][j]
            minimum = dungeon[i][j+1] if dungeon[i][j+1] < minimum else minimum
            right = self.calculateMinimumHPRec(copy.deepcopy(dungeon),i,j+1,minimum)
            dungeon[i][j+1] -= dungeon[i][j]

        minimum = prev_min

        if i < len(dungeon)-1:
            dungeon[i+1][j] += dungeon[i][j]
            minimum = dungeon[i+1][j] if dungeon[i+1][j] < minimum else minimum
            bottom = self.calculateMinimumHPRec(copy.deepcopy(dungeon),i+1,j,minimum)
            dungeon[i+1][j] -= dungeon[i][j]

        if i == len(dungeon)-1 and j == len(dungeon[0])-1:
            return minimum

        return max(right,bottom)

# 175 Combine Two Tables
SELECT P.FirstName, P.LastName,A.City,A.State
    FROM Person AS P, Address AS A
    WHERE P.PersonId = A.PersonId;

# 176 Second Highest Salary
SELECT (SELECT MAX(Salary) FROM Employee WHERE Salary NOT IN (SELECT MAX(Salary) FROM Employee)) SecondHighestSalary;

# 177 Nth Highest Salary
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
     # Write your MySQL query statement below.
     SELECT MAX(Salary) /*This is the outer query part */
            FROM Employee Emp1
            WHERE (N-1) = ( /* Subquery starts here */
                 SELECT COUNT(DISTINCT(Emp2.Salary))
                        FROM Employee Emp2
                        WHERE Emp2.Salary > Emp1.Salary)
  );
END

# 178 Rank Scores
SELECT Ranks.Score, Ranks.Rank FROM Scores LEFT JOIN
       ( SELECT r.Score, @curRow := @curRow + 1  Rank
            FROM (SELECT DISTINCT(Score), (SELECT @curRow := 0)
                      FROM Scores ORDER by Score DESC) r
       ) Ranks
       ON Scores.Score = Ranks.Score
       ORDER by Score DESC

# 179 Largest Number
class Solution:
    def largestNumber(self, nums):
        return ''.join(str(i) for i in sorted(nums, key=str, reverse=True))

# 180 Consecutive Numbers
SELECT DISTINCT(Num) AS ConsecutiveNums
FROM (
    SELECT
    Num,
    @counter := IF(@prev = Num, @counter + 1, 1) AS how_many_cnt_in_a_row,
    @prev := Num
    FROM Logs y, (SELECT @counter:=1, @prev:=NULL) vars
) sq
WHERE how_many_cnt_in_a_row >= 3

# 181 E ployees Earning More Than Their Managers


# 182 Duplicate Emails


# 183 Customers Who Never Order


# 184 Department Highest Salary


# 185 Department Top Three Salaries


# 186 Reverse Words in a String II


# 187 Repeated DNA Sequences


# 188 Best Time to Buy and Sell Stock IV


# 189 Rotate Array


# 190 Reverse Bits


# 191 N mber of 1 Bits


# 19 Word Frequency


# 193 Valid Phone Numbers


# 194 Transpose File


# 195 Tenth Line


# 196 Delete Duplicate Emails


# 197 Rising Temperature


# 198 House Robber


# 199 Binary Tree Right Side View


# 200 Number of Islands
class Solution:
    def numIslands(self, grid):
        forest = {}
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] and (i,j) not in forest:
                    forest[(i,j)] = None
                    self.DFS(i,j,forest,grid)
        return sum([1 for i in forest.values() if not i])

    def DFS(self,i,j,forest,grid):
        #doing 'if grid[i][j+1] does not check if you are out of bounds, only if the number that is supposed to be there is 0 or not
        #to check bounds, you have to do a j < len(grid[0])-1
        if j < len(grid[0])-1 and grid[i][j+1] and (i,j+1) not in forest:
            forest[(i,j+1)] = (i,j)
            self.DFS(i,j+1,forest,grid)
        if i < len(grid)-1 and grid[i+1][j] and (i+1,j) not in forest:
            forest[(i+1,j)] = (i,j)
            self.DFS(i+1,j,forest,grid)
