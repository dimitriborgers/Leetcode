# Q1 Two Sum
class Solution:
    def twoSum(self, nums, target):
        counter = {}
        for pos,e in enumerate(nums):
            if e in counter:
                return counter[e],pos
            else:
                complement = target - e
                counter[complement] = pos
        return None

# Q2 Add Two Numbers
class Solution:
    def addTwoNumbers(self, l1, l2):
        dummy = cur = ListNode(0)
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            cur.next = ListNode(carry%10)
            cur = cur.next
            carry //= 10
        return dummy.next

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
        return maxCount

# Q4 Median of Two Sorted Arrays
class Solution:
    def median(self, A, B):
        m, n = len(A), len(B)
        if m > n:
            A, B, m, n = B, A, n, m
        if n == 0:
            #Don't need to write a print statement
            raise ValueError

        #You're not just dividing both sides by 2
        imin, imax, half_len = 0, m, (m + n + 1) // 2
        while imin <= imax:
            #anything at index i or greater is closeParan part
            #openParan part always equal or bigger than closeParan part
            i = (imin + imax) // 2
            j = half_len - i
            if i < m and B[j-1] > A[i]:
                imin = i + 1
            elif i > 0 and A[i-1] > B[j]:
                imax = i - 1
            else:
                if i == 0:
                    max_of_openParan = B[j-1]
                elif j == 0:
                    max_of_openParan = A[i-1]
                else:
                    max_of_openParan = max(A[i-1], B[j-1])

                if (m + n) % 2 == 1:
                    return max_of_openParan

                if i == m:
                    min_of_closeParan = B[j]
                elif j == n:
                    min_of_closeParan = A[i]
                else:
                    min_of_closeParan = min(A[i], B[j])

                return (max_of_openParan + min_of_closeParan) / 2

# Q5 Longest Palindromic Substring
class Solution:
    def longestPalindrome(self, s):
        start = end = 0
        for i in range(len(s)):
            temp1 = self._utility_helper(s,i,i)
            temp2 = self._utility_helper(s,i,i+1)
            maxCount = max(temp1,temp2)
            if (maxCount > end - start):
                #take away 1 because for an even length string
                start = i - (maxCount - 1) // 2;
                end = i + maxCount // 2;
        return s[start:end+1]

    def _utility_helper(self,s,openParan,closeParan):
        while openParan >= 0 and closeParan < len(s) and s[openParan] == s[closeParan]:
            openParan -= 1
            closeParan += 1
        return closeParan - openParan - 1

# Q6 ZigZag Conversion
class Solution:
    def convert(self, s, numRows):
        rowList = [[] for _ in range(numRows)]
        increasing = True
        count = 0

        for i in range(len(s)):
            if count == numRows-1:
                increasing = False
            if count == 0:
                increasing = True

            if increasing == True:
                rowList[count].append(s[i])
                count += 1
            else:
                rowList[count].append(s[i])
                count -= 1

        return ''.join([''.join(i) for i in rowList])

# Q7 Reverse Integer
class Solution:
    def reverse(self, x):
        negative = True if x < 0 else False

        x = abs(x)
        outcome = 0
        while x != 0:
            pop = x % 10
            x = x // 10
            outcome = outcome*10 + pop

        if outcome > 2**31-1:
            return 0
        return outcome if negative == False else -outcome

# Q8 String to Integer (atoi)
class Solution:
    def myAtoi(self, s):
        ls = list(s.strip())
        if not ls:
            return 0

        sign = -1 if ls[0] == '-' else 1
        if ls[0] in ['-','+']:
            del ls[0]
        ret, i = 0, 0
        while i < len(ls) and ls[i].isdigit():
            ret = ret*10 + ord(ls[i]) - ord('0')
            i += 1
        return max(-2**31, min(sign * ret,2**31-1))

# Q9 Palindrome Number
class Solution:
    def isPalindrome(self, x):
        reverted = 0
        if x < 0:
            return False
        while x > reverted:
            pop = x % 10
            x = x // 10
            reverted = reverted * 10 + pop
        return True if reverted == x or reverted//10 == x else False

# Q10 Regular Expression Matching
class Solution:
    def isMatch(self, text, pattern):
        if not pattern:
            #if there is no pattern and no text, this returns True
            return not text

        #make sure there is still text, then check pattern.
        #and statement specifications:
            #True and False -> False
            #True and True -> True
            #False and True -> False
            #False and 5, 5 and False -> False
            #True and 5, 5 and True -> 5
            #5 and 'hello' -> 'hello'
        #or statement specifications:
            #True and 5, 5 and True -> True
            #False and 5, 5 and False -> 5
            #5 or 'Hello' -> 'Hello'
        #In this case, if you don't have bool(text), then first_match would equal whatever string text is if statement on right equals True.
        first_match = bool(text) and pattern[0] in (text[0], '.')

        #* means 0 or more, that's why would try pattern[2:]
        if len(pattern) >= 2 and pattern[1] == '*':
            return (self.isMatch(text, pattern[2:]) or
                    first_match and self.isMatch(text[1:], pattern))
        else:
            return first_match and self.isMatch(text[1:], pattern[1:])

# Q11 Container With Most Water
class Solution:
    def maxArea(self, height):
        i = 0
        j = len(height) - 1
        maxArea = 0

        while i != j:
            side = min(height[i],height[j])
            width = j - i
            maxArea = max(side*width, maxArea)
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return maxArea

# Q12 Integer to Roman
class Solution:
    def intToRoman(self, num):
        values = [ 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 ]
        numerals = [ "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" ]
        res = ""
        for i, v in enumerate(values):
            res += (num//v) * numerals[i]
            num %= v
        return res

# Q13 Roman to Integer
class Solution:
    def romanToInt(self, s):
        roman = {'M': 1000,'D': 500 ,'C': 100,'L': 50,'X': 10,'V': 5,'I': 1}
        z = 0
        for i in range(0, len(s) - 1):
            if roman[s[i]] < roman[s[i+1]]:
                z -= roman[s[i]]
            else:
                z += roman[s[i]]
        return z + roman[s[-1]]

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
        output = set()

        for i in range(len(nums)-2):
            j = i + 1
            end = len(nums)-1

            if i and nums[i] == nums[i - 1]:
                continue

            while j != end:
                #can't do sum(x,y,z), but can do sum([x,y,z])
                total = nums[i]+nums[j]+nums[end]

                if total < 0:
                    j += 1
                elif total > 0:
                    end -= 1
                else:
                    output.add((nums[i],nums[j],nums[end]))
                    j += 1

        return list(output)

# Q16 3Sum Closest
class Solution:
    def threeSumClosest(self, nums, target):
        nums.sort()
        end = len(nums)-1
        closest = float('inf')

        for i in range(len(nums)-2):
            j = i + 1

            if i and nums[i] == nums[i - 1]:
                continue

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
        if not digits:
            return []
        nums = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
        inputDigits = []
        #Dynamically creating variables can be done with list or dictionary
        for i in digits:
            inputDigits.append(nums[i])
        #you can't unpack a dictionary with **inputDigits with product()
        return list(''.join(i) for i in itertools.product(*inputDigits))

# Q18 4Sum
import collections

class Solution:
    def fourSum(self, nums, target):
        #in defaultdict(list), if key not found, automatically creates list
        nums, result, lookup = sorted(nums), [], collections.defaultdict(list)
        for i in range(len(nums)-1):
            for j in range(i + 1, len(nums)):
                #Can't remove boolean because appending must be done outside below for loop existence
                is_duplicated = False
                for x, y in lookup[nums[i] + nums[j]]:
                    if nums[x] == nums[i]:
                        is_duplicated = True
                        break
                if not is_duplicated:
                    lookup[nums[i] + nums[j]].append([i, j])
        #this creates a dictionary, not a set
        ans = {}
        for c in range(2, len(nums)):
            for d in range(c+1, len(nums)):
                if target - nums[c] - nums[d] in lookup:
                    for [a, b] in lookup[target - nums[c] - nums[d]]:
                        #check b < c to make sure you don't have duplicates of the same value in an answer
                        if b < c:
                            quad = [nums[a], nums[b], nums[c], nums[d]]
                            quad_hash = " ".join(str(quad))
                            if quad_hash not in ans:
                                ans[quad_hash] = True
                                result.append(quad)
        return result

# Q19 Remove Nth Node From End of List
class Solution:
    def removeNthFromEnd(self, head, n):
        dummy = ListNode(-1)
        dummy.next = head
        slow, fast = dummy, dummy

        for i in range(n):
            fast = fast.next

        while fast.next:
            slow, fast = slow.next, fast.next

        slow.next = slow.next.next

        return dummy.next

if __name__ == "__main__":
    head = ListNode(1)
    head.next = ListNode(2)
    head.next.next = ListNode(3)
    head.next.next.next = ListNode(4)
    head.next.next.next.next = ListNode(5)

    print(Solution().removeNthFromEnd(head, 2))

# Q20 Valid Parentheses
class Solution:
    def isValid(self, s):
        S = []
        for i in s:
            if i in '{[(':
                S.append(i)
            else:
                if not S:
                    return False
                x = S.pop()
                if i == ']' and x == '[':
                    continue
                elif i == ')' and x == '(':
                    continue
                elif i == '}' and x == '{':
                    continue
                else:
                    return False
        return len(S) == 0

# Q21 Merge Two Sorted Lists
class Solution:
    def mergeTwoLists(self, l1, l2):
        temp = None
        if l1 is None:
            return l2
        if l2 is None:
            return l1
        if l1.val <= l2.val:
            #since you assign temp to l1, you actually change the structure of l1 list itself
            #May need to use deepcopy if you don't want to alter original lists
            temp = l1
            temp.next = self.mergeTwoLists(l1.next, l2)
        else:
            temp = l2
            temp.next = self.mergeTwoLists(l1, l2.next)
        return temp

# Q22 Generate Parentheses
class Solution:
    def generateParenthesis(self, N):
        ans = []
        def backtrack(S = '', openParan = 0, closeParan = 0):
            if len(S) == 2 * N:
                ans.append(S)
                return
            if openParan < N:
                backtrack(S+'(', openParan+1, closeParan)
            if closeParan < openParan:
                backtrack(S+')', openParan, closeParan+1)

        backtrack()
        return ans

# Q23 Merge k Sorted Lists
class Solution:
    def mergeTwo(self,head1, head2):
        temp = None
        if head1 is None:
            return head2
        if head2 is None:
            return head1
        if head1.val <= head2.val:
            #not actually needed to do deepcopy since you don't care about changing original lists
            temp = copy.deepcopy(head1)
            temp.next = self.mergeTwo(head1.next, head2)
        else:
            temp = copy.deepcopy(head2)
            temp.next = self.mergeTwo(head1, head2.next)
        return temp

    def mergeKLists(self,lists):
        for i in range(len(lists)-1):
            temp = self.mergeTwo(lists[i],lists[i+1])
            lists[i+1] = temp
        return lists[-1]

class Solution1:
    def mergeKLists(self, lists):
        amount = len(lists)
        interval = 1
        while interval < amount:
            #amount-interval let's you make sure you don't go out of bounds
            for i in range(0, amount - interval, interval * 2):
                lists[i] = self.merge2Lists(lists[i], lists[i + interval])
            interval *= 2
        return lists[0] if amount > 0 else lists

    #non-recursive way of merging two lists
    def merge2Lists(self, l1, l2):
        head = point = ListNode(0)
        while l1 and l2:
            if l1.val <= l2.val:
                point.next = l1
                l1 = l1.next
            else:
                point.next = l2
                l2 = l1
                l1 = point.next.next
            point = point.next
        if not l1:
            point.next=l2
        else:
            point.next=l1
        return head.next

# Q24 Swap Nodes in Pairs
class Solution:
    #seq can't equal head because that will just create a temporary pass by reference, so changes won't be saved.
    def swapPairs(self, seq):
        current = seq.head
        while current and current.next:
            first = current
            second = current.next
            third = current.next.next

            current.next = third
            second.next = first

            if first == seq.head:
                seq.head = second
            else:
                old_first.next = second

            if third is None:
                return seq.head

            old_first = current
            current = third

# Q25 Reverse Nodes in k-Group
class Solution:
    def reverseKGroup(self, head, k):
        dummy = ListNode(-1)
        dummy.next = head

        cur, cur_dummy = head, dummy
        length = 0

        while cur:
            next_cur = cur.next
            length = (length + 1) % k

            if length == 0:
                next_dummy = cur_dummy.next
                self.reverse(cur_dummy, cur.next)
                cur_dummy = next_dummy

            cur = next_cur

        return dummy.next

    def reverse(self, begin, end):
            first = begin.next
            cur = first.next

            while cur != end:
                first.next = cur.next
                cur.next = begin.next
                begin.next = cur
                cur = first.next

if __name__ == "__main__":
    head = ListNode(1)
    head.next = ListNode(2)
    head.next.next = ListNode(3)
    head.next.next.next = ListNode(4)
    head.next.next.next.next = ListNode(5)
    print(Solution().reverseKGroup(head, 2))

# Q26 Remove Duplicates from Sorted Array
class Solution:
    def removeDuplicates(self, nums):
        for i in range(len(nums)):
            while i<len(nums) and nums[i] == nums[i-1]:
                del nums[i]
        return nums

# Q27 Remove Element
class Solution:
    def removeElement(self, nums, val):
        for i in range(len(nums)):
            while i<len(nums) and nums[i] == val:
                #can't do del if use for i in nums
                del nums[i]
        return nums

# Q28 Implement strStr()
class Solution:
    def strStr(self, haystack, needle):
        if haystack == '':
            return 0
        for i in range(len(haystack)):
            index = 0
            j = i+1
            if haystack[i] == needle[index]:
                index += 1
                #order matters in while statement
                while j < len(haystack) and haystack[j] == needle[index]:
                    if index == len(needle)-1:
                        return i
                    j += 1
                    index += 1
        return -1

import re
class Solution:
    def strStr2(self, haystack, needle):
        try:
            return re.search(needle,haystack).start()
        except:
            return -1

# Q29 Divide Two Integers
class Solution:
    def divide(self, dividend, divisor):
        #can't do xor
        isNegative = True if dividend < 0 or divisor < 0 else False
        count = 0
        dividend, divisor = abs(dividend),abs(divisor)
        while dividend > divisor:
            dividend -= divisor
            count += 1
        if isNegative:
            return -count
        return count

# Q30 Substring with Concatenation of All Words
import collections

class Solution:
    def findSubstring(self, s, words):
        result, m, n, k = [], len(s), len(words), len(words[0])
        if m < n*k:
            return result

        lookup = collections.defaultdict(int)
        for i in words:
            lookup[i] += 1

        for i in range(m+1-k*n):
            cur, j = collections.defaultdict(int), 0
            while j < n:
                #s[a:b] does not include b
                word = s[i+j*k:i+j*k+k]
                if word not in lookup:
                    break
                cur[word] += 1
                if cur[word] > lookup[word]:
                    break
                j += 1
            if j == n:
                result.append(i)

        return result

class Solution2:
    def findSubstring(self, s, words):
        result, m, n, k = [], len(s), len(words), len(words[0])
        if m < n*k:
            return result

        for i in range(m+1-k*n):
            setwords = set(words)
            counter = i
            temp = s[counter:counter+k]
            while setwords:
                if temp in setwords:
                    setwords.remove(temp)
                    counter += k
                    temp = s[counter:counter+k]
                else:
                    break
            else:
                result.append(i)
        return result

# Q31 Next Permutation
# if you have a for-loop with an else statement, the else statement is not triggered if you break out of the loop
class Solution:
    def nextPermutation(self, nums):
        right = len(nums)-1
        while nums[right] <= nums[right-1] and right-1 >=0:
            right -= 1
        if right == 0:
            return self.reverse(nums,0,len(nums)-1)
        pivot = right-1
        successor = 0
        for i in range(len(nums)-1,pivot,-1):
            if nums[i] > nums[pivot]:
                successor = i
                break
        nums[pivot],nums[successor] = nums[successor],nums[pivot]
        self.reverse(nums,pivot+1,len(nums)-1)

    def reverse(self,nums,l,r):
        while l < r:
            nums[l],nums[r] = nums[r],nums[l]
            l += 1
            r -= 1

# Q32 Longest Valid Parentheses
class Solution:
    def longestValidParentheses(self, s):
        stack = [-1]
        length = 0
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    #better than doing, length = i - stack[-1] if i - stack[-1] > length else length
                    length = max(length, i - stack[-1])
        return length

# Q33 Search in Rotated Sorted Array
class Solution:
    def search(self, nums, target):
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                return mid
            elif (nums[left] <= target < nums[mid]) or (nums[mid] < nums[left] and not (nums[mid] < target <= nums[right])):
                right = mid - 1
            else:
                left = mid + 1
        return -1

# Q34 Find First and Last Position of Element in Sorted Array
class Solution:
    def searchRange(self, nums, target):
        left,right = 0,len(nums)-1

        while left <= right:
            mid = left + (right-left) // 2

            if nums[mid] == target:
                i = j = 0
                while nums[mid-i] == target:
                    i += 1
                while nums[mid+j] == target:
                    j += 1
                return (mid-(i-1),mid+(j-1))
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return (-1,-1)

# Q35 Search Insert Position
class Solution:
    #Using binary search would be faster
    def searchInsert(self, nums, target):
        for i in range(len(nums)):
            if nums[i] > target:
                return i
            elif nums[i] == target:
                return i
            else:
                continue
        return len(nums)

# Q36 Valid Sudoku
class Solution:
    def isValidSudoku(self, board):
        for i in range(len(board)):
            if not self.isValid([board[i][j] for j in range(len(board))]) or not self.isValid([board[j][i] for j in range(len(board))]):
                return False

        for i in range(3):
            for j in range(3):
                if not self.isValid([board[m][n] for n in range(3 * j, 3 * j + 3) for m in range(3 * i, 3 * i + 3)]):
                    return False
        return True

    def isValid(self,row):
        row = [i for i in row if i != '.']
        if len(set(row)) == len(row):
            return True

# Q37 Sudoku Solver
class Solution:
    def solveSudoku(self, board):
        #Inside functions don't have self as argument or in front when they are called.
        def isValidSudoku(board):
            for i in range(len(board)):
                if not isValid([board[i][j] for j in range(len(board))]) or not isValid([board[j][i] for j in range(len(board))]):
                    return False

            for i in range(3):
                for j in range(3):
                    if not isValid([board[m][n] for n in range(3 * j, 3 * j + 3) for m in range(3 * i, 3 * i + 3)]):
                        return False
            return True

        def isValid(row):
            row = [i for i in row if i != '.']
            if len(set(row)) == len(row):
                return True

        def solver(board):
            for i in range(len(board)):
                for j in range(len(board)):
                    if(board[i][j] == '.'):
                        for k in range(9):
                            #how to get a string of an int
                            board[i][j] = chr(ord('1') + k)
                            #solver(board) cannot be deepcopy of board
                            if isValidSudoku(board) and solver(board):
                                return True
                            board[i][j] = '.'
                        return False
            return True

# Q38 Count and Say
import itertools

class Solution:
    def countAndSay(self, n):
        if n == 1:
            return '1'
        if n == 2:
            return '11'

        tmp =  self.countAndSay(n-1)
        outcome = ''
        for value,times in itertools.groupby(tmp):
            outcome += str(len(list(times)))+value
        return outcome

# Q39 Combination Sum
class Solution:
    def combinationSum(self, candidates, target):
        result = []
        self.combinationSumRecu(sorted(candidates), result, 0, [], target)
        return result

    def combinationSumRecu(self, candidates, result, start, intermediate, target):
        if target == 0:
            #has to be list() because this creates a new intermediate that won't be alterned by procedures following addition
            result.append(list(intermediate))
        while start < len(candidates) and candidates[start] <= target:
            intermediate.append(candidates[start])
            self.combinationSumRecu(candidates, result, start, intermediate, target - candidates[start])
            intermediate.pop()
            start += 1

# Q40 Combination Sum II
class Solution:
    def combinationSum2(self, candidates, target):
        result = []
        self.combinationSumRecu(sorted(candidates), result, 0, [], target)
        return result

    def combinationSumRecu(self, candidates, result, start, intermediate, target):
        if target == 0 and sorted(intermediate) not in result:
            result.append(sorted(list(intermediate)))
        while start < len(candidates) and candidates[start] <= target:
            intermediate.append(candidates[start])
            self.combinationSumRecu(candidates, result, start + 1, intermediate, target - candidates[start])
            intermediate.pop()
            start += 1

# Q41 First Missing Positive
class Solution:
    def firstMissingPositive(self, A):
        i = 0
        while i < len(A):
            #This loop will only move elements between 0 and length of list. Therefore, you are putting the small elements in order at front
            if A[i] > 0 and A[i] - 1 < len(A) and A[i] != A[A[i]-1]:
                A[A[i]-1], A[i] = A[i], A[A[i]-1]
            else:
                i += 1

        for i, integer in enumerate(A):
            if integer != i + 1:
                return i + 1
        return len(A) + 1

# Q42 Trapping Rain Water
class Solution:
    def trap(self, height):
        left, right = 0, len(height)-1
        ans = 0
        left_max = right_max = 0

        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    ans += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    ans += right_max - height[right]
                right -= 1
        return ans

# Q43 Multiply Strings
class Solution:
    def multiply(self, num1, num2):

        num1, num2 = num1[::-1], num2[::-1]

        #multiplying two numbers together always creates a result that is length of length of two numbers
        res = [0] * (len(num1) + len(num2))
        for i in range(len(num1)):
            for j in range(len(num2)):
                res[i + j] += int(num1[i]) * int(num2[j])

                #this is remainder being carried over
                res[i + j + 1] += res[i + j] // 10
                res[i + j] %= 10

        # Skip leading 0s.
        i = len(res) - 1
        while i > 0 and res[i] == 0:
            i -= 1

        return ''.join(str(x) for x in res[i::-1])

# Q44 Wildcard Matching
class Solution:
    def isMatch(self, s, p):
        if not p or not s:
            return not s and not p
        if p[0] != '*':
            if p[0] == s[0] or p[0] == '?':
                return self.isMatch(s[1:], p[1:])
            else:
                return False
        else:
            while len(s) > 0:
                #p[1:] does not change for every loop since it returns False everytime until breaks out
                if self.isMatch(s, p[1:]):
                    return True
                s = s[1:]
            return self.isMatch(s, p[1:])

# Q45 Jump Game II
class Solution:
    def jump(self, A):
        jump_count = 0
        reachable = 0
        curr_reachable = 0
        for i, length in enumerate(A):
            if i > reachable:
                return -1
            if i > curr_reachable:
                curr_reachable = reachable
                jump_count += 1
            reachable = max(reachable, i + length)
        return jump_count

# Q46 Permutations
import itertools

class Solution:
    def permute(self, nums):
        return list(itertools.permutations(nums,len(nums)))

# Q47 Permutations II
import itertools

class Solution:
    def permute(self, nums):
        return set(list(itertools.permutations(nums,len(nums))))

# Q48 Rotate Image
# Sort of In place
class Solution1:
    def rotate(self, matrix):
        #Difference between nums[:] = and nums = is that the latter doesn't replace elements in the original list.
        #matrix[:] = zip(*matrix[::-1]) works as well
        matrix[::] = zip(*matrix[::-1])
        '''
        >>> a = list(range(10))
        >>> b = a
        >>> a[:] = [0, 0, 0] # changes what a and b both refer to
        >>> b
        [0, 0, 0]

        >>> a = list(range(10))
        >>> b = a
        >>> a = [0, 0, 0] # a now refers to a different list than b
        >>> b
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        '''

# 100% In place
class Solution2:
    def rotate(self, A):
        n = len(A)
        for i in range(n//2):
            for j in range(n-n//2):
                #complement operator
                #~0 -> -1
                #~1 -> -2
                #~-5 -> 4
                #using ~ in matrix indexing will give you opposite end of indexes given.
                A[i][j], A[~j][i], A[~i][~j], A[j][~i] = A[~j][i], A[~i][~j], A[j][~i], A[i][j]

# Q49 Group Anagrams
class Solution:
    def groupAnagrams(self, strs):
        ans = collections.defaultdict(list)
        for s in strs:
            ans[tuple(sorted(s))].append(s)

        return list(ans.values())

# Q50 Pow(x, n)
class Solution:
    def myPow(self,x,n):
        if n == 0:
            return 1.0
        elif n > 0:
            result = self.Pow(x,n)
        else:
            result = 1/self.Pow(x,-n)
        return result

    def Pow(self,x,n):
        if n == 1:
            return x
        partial = self.Pow(x,n//2)
        total = partial * partial
        if n % 2 != 0:
            total *= x
        return total
