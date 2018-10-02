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
                self.__n4 = read4(self.__buf4)  # Read more characters.
                if self.__n4:
                    self.__i4 = 0
                else:  # Buffer has been empty.
                    break

        return i

if __name__ == "__main__":
    global file_content
    sol = Solution()
    buf = ['' for _ in xrange(100)]
    file_content = "ab"
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
            output = len(lists) if len(lists) > output else output
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


# 162 Find Peak Element


# 163 Missing Ranges


# 164 Maximum Gap


# 165 Compare Version Numbers


# 166 Fraction to Recurring Decimal


# 167 Two Sum II - Input array is sorted


# 168 Excel Sheet Column Title


# 169 Majority Element


# 170 Two Sum III - Data structure design


# 171 E cel Sheet Column Number


# 172 Factorial Trailing Zeroes


# 173 Binary Search Tree Iterator


# 174 Dungeon Game


# 175 Combine Two Tables


# 176 Second Highest Salary


# 177 Nth Highest Salary


# 178 Rank Scores


# 179 Largest Number


# 180 Consecutive Numbers


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

