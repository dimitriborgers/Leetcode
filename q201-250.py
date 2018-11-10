# Q201 Bitwise AND of Numbers Range


# Q202 Happy Number


# Q203 Remove Linked List Elements


# Q204 Count Primes


# Q205 Isomorphic Strings
import collections

class Solution:
    def isIsomorphic(self, s, t):
        seq = collections.defaultdict(set)
        for i in range(len(s)):
            seq[s[i]].add(t[i])
            if len(seq[s[i]]) > 1:
                return False
        return True

# Q206 Reverse Linked List


# Q207 Course Schedule


# Q208 Implement Trie (Prefix Tree)


# Q209 Minimum Size Subarray Sum


# Q210 Course Schedule II


# Q211 Add and Search Word - Data structure design


# Q212 Word Search II


# Q213 House Robber II


# Q214 Shortest Palindrome


# Q215 Kth Largest Element in an Array


# Q216 Combination Sum III


# Q217 Contains Duplicate
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

# Q218 The Skyline Problem
import heapq

class Solution:
    #When going through buildings, find all critical points (coordinates of where building starts/ends) and label each with start or end of building, so you should just have x,h for each point. Always take the top of the building for both starters and enders heights.
    #For starters, if starter_height > heap.max(), add x,y to skyline, then add x,y to heap
    #For enders, remove ender_height from heap. If heap.max() changes, add x,heap.max() to skyline
    #for each critical point c:
    #for each rectangle r above c (not including the right edge of rectangles):
    #c.y gets the max of r.height and the previous value of c.y
    #if two buildings start at same spot, iterate through taller one first
    #if two buildings end at same point, iterate through the lower one first
    #if two buildings are side-by-side, then the next building start should be iterated through before the end of the first building
    def getSkyline(self, buildings):
        #list.sort(key=lambda x: (x[0], x[2]))
        #This sorts list by x[0] first, and then x[2] where x[0] is the same
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
            #Two numbers with an 'and' in between
            #Left of 'and' must evaluate to True
                #if it evaluates to True, then variable is assigned value of right of 'and'
                #if it evaluates to False, then variable is assigned False
            height = len(heapBuildings) and -heapBuildings[0][0]
            if len(skyline) == 0 or skyline[-1][1] != height:
                skyline.append([start, height])
        return skyline

# Q219 Contains Duplicate II
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

# Q220 Contains Duplicate III
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

# Q221 Maximal Square


# Q222 Count Complete Tree Nodes
class Solution:
    #complete binary tree has 2^depth - 1 nodes
    def height(self,root):
        return -1 if not root else 1 + self.height(root.left)

    def countNodes(self,root):
        nodes = 0
        h = self.height(root)
        while root:
            if self.height(root.right) == h - 1:
                #same thing as: nodes += 2**h
                nodes += 1 << h
                root = root.right
            else:
                #same thing as: nodes += 2**(h-1)
                nodes += 1 << (h - 1)
                root = root.left
            h -= 1
        return nodes

# Q223 Rectangle Area


# Q224 Basic Calculator


# Q225 Implement Stack using Queues


# Q226 Convert Binary Tree


# Q227 Basic Calculator II


# Q228 Summary Ranges


# Q229 Majority Element II


# Q230 Kth Smallest Element in a BST


# Q231 Power of Two


# Q232 Implement Queue using Stacks


# Q233 Number of Digit One


# Q234 Palindrome Linked List


# Q235 Lowest Common Ancestor of a Binary Search Tree


# Q236 Lowest Common Ancestor of a Binary Tree


# Q237 Delete Node in a Linked List


# Q238 Product of Array Except Self


# Q239 Sliding Window Maximum


# Q240 Search a 2D Matrix II

#2Q41 Different Ways to Add Parentheses


# Q242 Valid Anagram


# Q243 Shortest Word Distance


# Q244 Shortest Word Distance II


# Q245 Shortest Word Distance III


# Q246 Strobogrammatic Number


# Q247 Strobogrammatic Number II
class Solution:
    lookup = {'0':'0', '1':'1', '6':'9', '8':'8', '9':'6'}

    def findStrobogrammatic(self, n):
        return self.findStrobogrammaticRecu(n, n)

    def findStrobogrammaticRecu(self, n, k):
        if k == 0:
            return ['']
        elif k == 1:
            return ['0', '1', '8']

        result = []
        for num in self.findStrobogrammaticRecu(n, k - 2):
            for key, val in self.lookup.items():
                if n != k or key != '0':
                    result.append(key + num + val)

        return result

# Q248 Strobogrammatic Number III
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

# Q249 Group Shifted Strings


# Q250 Count Univalue Subtrees


