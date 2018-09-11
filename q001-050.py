# Q1 Two Sum
class Solution:
    def twoSum(self, nums, target):
        counter = {}
        for i in nums:
            if i in counter:
                return i,counter[i]
            else:
                complement = target - i
                counter[complement] = i
        return None

# Q2 Add Two Numbers
import linked_list_mod as ll
class LinkedList:

    class ListNode:
        __slots__ = 'value','next'

        def __init__(self, _val, _next):
            self.value = _val
            self.next = _next

    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def __len__(self):
        return self.size

    def __repr__(self):
        if self.size == 0:
            return '[]'
        outcome = []
        current = self.head
        while current:
            outcome.append(current.value)
            current = current.next
        return "{}".format(outcome)

    def addHead(self,value):
        if self.head == None:
            self.head = self.tail = self.ListNode(value, self.head)
        else:
            self.head = self.ListNode(value, self.head)
        self.size += 1

    def addTail(self,value):
        if self.head == None:
            self.head = self.tail = self.ListNode(value, self.head)
        else:
            self.tail.next = self.ListNode(value,None)
            self.tail = self.tail.next
        self.size += 1

    def removeHead(self):
        old_head = self.head
        self.head = self.head.next
        return old_head

    def removeTail(self):
        current = self.head
        while current.next.next:
            current = current.next
        old_head = current.next
        self.tail = current
        self.tail.next = None
        return old_head

class Solution:

    def addTwoNumbers(self, l1, l2):
        l3 = LinkedList()
        remainder = 0
        overflow = 0
        while l1.head:
            temp1 = l1.removeHead()
            temp2 = l2.removeHead()

            if remainder == 0:
                if temp1.value+temp2.value > 9:
                    if not temp1.next:
                        overflow = 1
                    remainder = 1
                    addition = temp1.value+temp2.value-10
                else:
                    addition = temp1.value+temp2.value
            else:
                if temp1.value+temp2.value+remainder > 9:
                    if not temp1.next:
                        overflow = 1
                    remainder = 1
                    addition = temp1.value+temp2.value+remainder-10
                else:
                    addition = temp1.value+temp2.value+remainder

            l3.addTail(addition)
            if overflow == 1:
                l3.addTail(1)
            print(l3)

# Q3 Longest Substring Without Repeating Characters
class Solution:
    def lengthOfLongestSubstring(self, s):
        i = j = 0
        seq = set()
        maxCount = 0
        count = 0
        while i < len(s) and j < len(s):
            if s[j] not in seq:
                seq.add(s[j])
                j += 1
                count = len(seq)
                if count > maxCount:
                    maxCount = count
            else:
                seq.remove(s[i])
                i += 1
        print(maxCount)

# Q4 Median of Two Sorted Arrays
class Solution:
    def median(self, A, B):
    m, n = len(A), len(B)
    if m > n:
        A, B, m, n = B, A, n, m
    if n == 0:
        raise ValueError

    imin, imax, half_len = 0, m, (m + n + 1) / 2
    while imin <= imax:
        i = (imin + imax) / 2
        j = half_len - i
        if i < m and B[j-1] > A[i]:
            # i is too small, must increase it
            imin = i + 1
        elif i > 0 and A[i-1] > B[j]:
            # i is too big, must decrease it
            imax = i - 1
        else:
            # i is perfect

            if i == 0: max_of_left = B[j-1]
            elif j == 0: max_of_left = A[i-1]
            else: max_of_left = max(A[i-1], B[j-1])

            if (m + n) % 2 == 1:
                return max_of_left

            if i == m: min_of_right = B[j]
            elif j == n: min_of_right = A[i]
            else: min_of_right = min(A[i], B[j])

            return (max_of_left + min_of_right) / 2.0

# Q5 Longest Palindromic Substring
class Solution:
    def longestPalindrome(self, s):
        start = end = 0
        for i in range(len(s)):
            temp1 = self._utility_helper(s,i,i)
            temp2 = self._utility_helper(s,i,i+1)
            maxCount = max(temp1,temp2)
            if (maxCount > end - start):
                start = i - (maxCount - 1) // 2;
                end = i + maxCount // 2;
        return s[start:end+1]

    def _utility_helper(self,s,left,right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

# Q6 ZigZag Conversion
class Solution:
    def convert(self, s, numRows):
        rowList = [[] for _ in range(numRows)]
        increasing = True
        count = 0

        for i in range(len(s)):
            if count == 3:
                increasing = False
            if count == 0:
                increasing = True

            if increasing == True:
                rowList[count].append(s[i])
                count += 1
            else:
                rowList[count].append(s[i])
                count -= 1
        outcome = []
        for i in rowList:
            for j in i:
                outcome.append(j)
        return ''.join(outcome)

# Q7 Reverse Integer
class Solution:
    def reverse(self, x):
        negative = True if x < 0 else False

        x = abs(x)
        outcome = 0
        while x != 0:
            pop = x % 10
            x = int(x/10)

            outcome = outcome*10 + pop
        return outcome if negative == False else -outcome

# Q8 String to Integer (atoi)
class Solution:
    def myAtoi(self, string):
        string = string.strip()
        string = list(string)
        outcome = 0
        if string[0] not in '-0123456789':
            return False
        else:
            negative = True if string[0] == '-' else False
            if negative:
                for i in range(1,len(string)):
                    if string[i] in '0123456789':
                        outcome = outcome * 10 + int(string[i])
                    else:
                        break
            else:
                for i in range(len(string)):
                    if string[i] in '0123456789':
                        outcome = outcome * 10 + int(string[i])
                    else:
                        break
        return outcome if not negative else -outcome

# Q9 Palindrome Number
class Solution:
    def isPalindrome(self, x):
        reverted = 0
        if x < 0:
            return False
        while x > reverted:
            pop = x % 10
            x = int(x / 10)
            reverted = reverted * 10 + pop
        return True if reverted == x or int(reverted/10) == x else False

# Q10 Regular Expression Matching
class Solution(object):
    def isMatch(self, text, pattern):
        dp = [[False] * (len(pattern) + 1) for _ in range(len(text) + 1)]

        dp[-1][-1] = True
        for i in range(len(text), -1, -1):
            for j in range(len(pattern) - 1, -1, -1):
                first_match = i < len(text) and pattern[j] in {text[i], '.'}
                if j+1 < len(pattern) and pattern[j+1] == '*':
                    dp[i][j] = dp[i][j+2] or first_match and dp[i+1][j]
                else:
                    dp[i][j] = first_match and dp[i+1][j+1]

        return dp[0][0]

# Q11 Container With Most Water
class Solution:
    def maxArea(self, height):
        i = 0
        j = len(height) - 1
        maxArea = 0

        while i != j:
            side = min(height[i],height[j])
            width = j - i
            maxArea = side*width if side*width > maxArea else maxArea
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return maxArea

# Q12 Integer to Roman
class Solution:
    def intToRoman(self, x):
        outcome = ''
        if x > 1000:
            quotient,x = divmod(x,1000)
            outcome += quotient*'M'
        if x > 100:
            quotient,x = divmod(x,100)
            if quotient == 9:
                outcome += 'CM'
            elif quotient >= 5:
                quotient %= 5
                outcome += 'D' + quotient*'C'
            elif quotient == 4:
                outcome += 'CD'
            else:
                outcome += quotient*'C'
        if x > 10:
            quotient,x = divmod(x,10)
            if quotient == 9:
                outcome += 'XC'
            elif quotient >= 5:
                quotient %= 5
                outcome += 'L' + quotient*'X'
            elif quotient == 4:
                outcome += 'XL'
            else:
                outcome += quotient*'X'
        if x < 10:
            if x == 9:
                outcome += 'IX'
            elif x >= 5:
                x %= 5
                outcome += 'V' + x*'I'
            elif x == 4:
                outcome += 'IV'
            else:
                outcome += x*'I'
        return outcome

# Q13 Roman to Integer
Same Q12

# Q14 Longest Common Prefix
class Solution:
    def longestCommonPrefix(self, strs):
        longest = strs[0]
        for i in range(1,len(strs)):
            tempL = ''
            for j in range(min(len(longest),len(strs[i]))):
                if longest[j] == strs[i][j]:
                    tempL += longest[j]
            longest = tempL
        return longest

# Q15 3Sum
class Solution:
    def threeSum(self, nums):
        nums.sort()
        end = len(nums)-1
        output = []
        for i in range(len(nums)-2):
            j = i + 1

            while j != end:
                total = nums[i]+nums[j]+nums[end]
                #can't do sum(x,y,z), but can do sum([x,y,z])

                if total < 0:
                    j += 1
                elif total > 0:
                    end -= 1
                else:
                    output.append([nums[i],nums[j],nums[end]])
                    j += 1

        unique_lst = []
        [unique_lst.append(sublst) for sublst in output if not unique_lst.count(sublst)]

        return unique_lst

# Q16 3Sum Closest
class Solution:
    def threeSumClosest(self, nums, target):
        nums.sort()
        end = len(nums)-1
        closest = 1000000

        for i in range(len(nums)-2):
            j = i + 1

            while j < end:
                total = nums[i]+nums[j]+nums[end]
                closest = total if abs(total-target) < abs(closest-target) else closest

                if total < target:
                    j += 1
                elif total > target:
                    end -= 1
                else:
                    return nums[i]+nums[j]+nums[end]
        return closest

# Q17 Letter Combinations of a Phone Number
import itertools

class Solution:
    def letterCombinations(self, digits):
        nums = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqr','8':'stu','9':'vwx'}
        inputDigits = []
        #Dynamically creating variables can be done with list or dictionary
        for i in digits:
            inputDigits.append(nums[i])
        #you can't unpack a dictionary with **inputDigits with product()
        return list(''.join(i) for i in itertools.product(*inputDigits))

# Q18 4Sum
import collections

class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        nums, result, lookup = sorted(nums), [], collections.defaultdict(list)
        print(nums)
        for i in range(len(nums)-1):
            for j in range(i + 1, len(nums)):
                print("first {}".format(lookup))
                is_duplicated = False
                for x, y in lookup[nums[i] + nums[j]]:
                    print("second {}".format(lookup))
                    if nums[x] == nums[i]:
                        is_duplicated = True
                        break
                if not is_duplicated:
                    lookup[nums[i] + nums[j]].append([i, j])
        ans = {}
        print('get here')
        for c in range(2, len(nums)):
            for d in range(c+1, len(nums)):
                if target - nums[c] - nums[d] in lookup:
                    for [a, b] in lookup[target - nums[c] - nums[d]]:
                        if b < c:
                            quad = [nums[a], nums[b], nums[c], nums[d]]
                            quad_hash = " ".join(str(quad))
                            if quad_hash not in ans:
                                ans[quad_hash] = True
                                result.append(quad)
        return result

        #s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
        # for k in s:
            # print(k,v)

# Q19 Remove Nth Node From End of List
class Solution:
    def removeNthFromEnd(self, head, n):
        current = head
        count = 0
        while count < n - 1:
            current = current.next
            count += 1
        old_next = current.next
        current.next = current.next.next
        old_next.next = None

# Q20 Valid Parentheses
class Solution:
    def isValid(self, s):
        S = []
        for i in s:
            if i in '{[(':
                S.append(i)
            else:
                x = S.pop()
                if i == ']' and x == '[':
                    continue
                elif i == ')' and x == '(':
                    continue
                elif i == '}' and x == '{':
                    continue
                else:
                    return False
        return True
