# 201 Bitwise AND of Numbers Range


# 202 Happy Number


# 203 Remove Linked List Elements


# 204 Count Primes


# 205 Isomorphic Strings
import collections

class Solution:
    def isIsomorphic(self, s, t):
        seq = collections.defaultdict(set)
        for i in range(len(s)):
            seq[s[i]].add(t[i])
            if len(seq[s[i]]) > 1:
                return False
        return True

# 206 Reverse Linked List


# 207 Course Schedule


# 208 Implement Trie (Prefix Tree)


# 209 Minimum Size Subarray Sum


# 210 Course Schedule II


# 211 Add and Search Word - Data structure design


# 21# 2 Word Search II


# 213 House Robber II


# 214 Shortest Palindrome


# 215 Kth Largest Element in an Array


# 216 Combination Sum III


# 217 Contains Duplicate
class Solution:
    def containsDuplicate1(self, nums):
        lookup = {}
        for i in nums:
            if i in lookup:
                return True
            lookup[i] = i
        return False

    def containsDuplicate2(self, nums):
        nums.sort()
        for i in range(1,len(nums)):
            if nums[i] == nums[i-1]:
                return True
        return False

# 218 The Skyline Problem
import heapq

class Solution:
    #When going through buildings, find all critical points (coordinates of where building starts/ends) and label each with start or end of building, so you should just have x,h for each point. Always take the top of the building for both starters and enders heights.
    #For starters, if starter_height > heap.max(), add x,y to skyline
    #For enders, remove ender_height from heap. If heap.max() changes, add x,heap.max() to skyline
    #for each critical point c:
    #for each rectangle r above c (not including the right edge of rectangles):
    #c.y gets the max of r.height and the previous value of c.y
    def getSkyline(self, buildings):
        buildings.sort()
        index, length = 0, len(buildings)
        heapBuildings, skyline = [], []
        while index < length or len(heapBuildings) > 0:
            if len(heapBuildings) == 0 or (index < length and buildings[index][0] <= -heapBuildings[0][1]):
                start = buildings[index][0]
                while index < length and buildings[index][0] == start:
                    heapq.heappush(heapBuildings, [-buildings[index][2], -buildings[index][1]])
                    index += 1
            else:
                start = -heapBuildings[0][1]
                while len(heapBuildings) > 0 and -heapBuildings[0][1] <= start:
                    heapq.heappop(heapBuildings)
            height = len(heapBuildings) and -heapBuildings[0][0]
            if len(skyline) == 0 or skyline[-1][1] != height:
                skyline.append([start, height])
        return skyline

# 219 Contains Duplicate II
class Solution:
    def containsNearbyDuplicate(self, nums, k):
        lookup = {}
        for i, num in enumerate(nums):
            if num not in lookup:
                lookup[num] = i
            else:
                if i - lookup[num] <= k:
                    return True
                lookup[num] = i
        return False

# 220 Contains Duplicate III
class Solution:
    #For a given element x, is there an item in the window that is within the range of [x-t, x+t]?
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        if t < 0 or not nums or k <= 0:
            return False
        buckets = {}
        #add 1 so that you don't divide by 0
        #width is just dividing up numbers into buckets
        width = t + 1

        for pos, element in enumerate(nums):
            bucket = element // width
            if bucket in buckets:
                return True
            else:
                buckets[bucket] = element
                if bucket - 1 in buckets and element - buckets[bucket-1] <= t:
                    return True
                if bucket + 1 in buckets and buckets[bucket+1] - element <= t:
                    return True
                if pos >= k:
                    del buckets[nums[pos-k] // width]
        return False

# 221 aximal Square


# 222 unt Complete Tree Nodes


# 223 rectangle Area


# 224 basic Calculator


# 225 implement Stack using Queues


# 226 convert Binary Tree


# 227 basic Calculator II


# 228 ummary Ranges


# 229 ajority Element II


# 230 Kth Smallest Element in a BST


# 231 Power of Two


# 23# 2 I plement Queue using Stacks


# 233 Number of Digit One


# 234 Palindrome Linked List


# 235 Lowest Common Ancestor of a Binary Search Tree


# 236 Lowest Common Ancestor of a Binary Tree


# 237 Delete Node in a Linked List


# 238 Product of Array Except Self


# 239 Sliding Window Maximum


# 240 Search a 2D Matrix II#


#241 Different Ways to Add Parentheses


# 242 Valid Anagram


# 243 Shortest Word Distance


# 244 Shortest Word Distance II


# 245 Shortest Word Distance III


# 246 Strobogrammatic Number


# 247 Strobogrammatic Number II


# 248 Strobogrammatic Number III
class Solution:
    lookup = {'0':'0', '1':'1', '6':'9', '8':'8', '9':'6'}
    cache = {}

    def strobogrammaticInRange(self, low, high):
        return self.countStrobogrammaticUntil(high, False) - self.countStrobogrammaticUntil(str(int(low)-1), False)

    def countStrobogrammaticUntil(self, num,  can_start_with_0):
        if can_start_with_0 and num in self.cache:
            return self.cache[num]

        count = 0
        if len(num) == 1:
            for c in ['0', '1', '8']:
                if num[0] >= c:
                    count += 1
            self.cache[num] = count
            return count

        #also works to write Solution.loopkup.items()
        for key, val in self.lookup.items():
            if can_start_with_0 or key != '0':
                if num[0] > key:
                    if len(num) == 2:  # num is like "21"
                        count += 1
                    else:  # num is like "201"
                        count += self.countStrobogrammaticUntil('9' * (len(num) - 2), True)
                elif num[0] == key:
                    if len(num) == 2:  # num is like 12".
                        if num[-1] >= val:
                            count += 1
                    else:
                        if num[-1] >= val:  # num is like "102".
                            count += self.countStrobogrammaticUntil(self.getMid(num), True);
                        elif (self.getMid(num) != '0' * (len(num) - 2)):  # num is like "110".
                            count += self.countStrobogrammaticUntil(self.getMid(num), True) - self.isStrobogrammatic(self.getMid(num))

        if not can_start_with_0: # Sum up each length.
            for i in range(len(num) - 1, 0, -1):
                count += self.countStrobogrammaticByLength(i)
        else:
            self.cache[num] = count

        return count

    def getMid(self, num):
        return num[1:len(num) - 1]

    def countStrobogrammaticByLength(self, n):
        if n == 1:
            return 3
        elif n == 2:
            return 4
        elif n == 3:
            return 4 * 3
        else:
            return 5 * self.countStrobogrammaticByLength(n - 2)

# 249 Group Shifted Strings


# 250 Count Univalue Subtrees


