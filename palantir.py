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

#------------------------------------------------------------------------------

# Q10 Regular Expression Matching
class Solution:
    def isMatch(self, text, pattern):
        if not pattern:
            #if there is no pattern and no text, this returns True
            return not text

        #make sure there is still text, then check pattern.
        first_match = text and pattern[0] in (text[0], '.')

        #* means 0 or more, that's why would try pattern[2:]
        if len(pattern) >= 2 and pattern[1] == '*':
            return (self.isMatch(text, pattern[2:]) or
                    first_match and self.isMatch(text[1:], pattern))
        else:
            return first_match and self.isMatch(text[1:], pattern[1:])

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

# Q42 Trapping Rain Water
class Solution1:
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

class Solution2:
    def trap(self, height):
        leftmax,rightmax = height[0],height[-1]
        lefti,righti = 0,len(height)-1
        total = 0

        while lefti < righti:
            if height[lefti] <= height[righti]:
                total = self._check(leftmax,rightmax,height,lefti,total)
                lefti += 1
                leftmax = max(leftmax,height[lefti])
            else:
                total = self._check(leftmax,rightmax,height,righti,total)
                righti -= 1
                rightmax = max(rightmax,height[righti])
        return total

    def _check(self,leftmax,rightmax,height,pos,total):
        if leftmax and rightmax and height[pos] < leftmax and height[pos] < rightmax:
            total += min(leftmax,rightmax) - height[pos]
        return total

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

# Q121 Best Time to Buy and Sell Stock
class Solution1:
    def maxProfit(self, prices):
        #how to create infinity
        max_profit, min_price = 0, float("inf")
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        return max_profit

class Solution2:
    def maxProfit(self, prices):
        localmin = localmax = prices[0]
        profit = 0

        for i in range(1,len(prices)):
            if localmin == localmax and prices[i] < localmin:
                localmin = localmax = prices[i]
            elif prices[i] > localmax:
                localmax = prices[i]
            elif prices[i] < localmin:
                localmin = localmax = prices[i]
            profit = max(profit,localmax-localmin)

        return profit

#------------------------------------------------------------------------------

# Q94 Binary Tree Inorder Traversal
class Solution:
    def inorderTraversalRec(self,root):
        self.result = []
        return self._inorder_helper(root)

    def _inorder_helper(self, root):
        if root.left:
            self._inorder_helper(root.left)
        #can't be append(root) because this would traverse from root every time when you print
        self.result.append(root.val)
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

#------------------------------------------------------------------------------

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

        return root

    def detectBroken(self, broken, pre, cur):
        if pre and pre.val > cur.val:
            if broken[0] is None:
                broken[0] = pre
            broken[1] = cur

#------------------------------------------------------------------------------

# Q122 Best Time to Buy and Sell Stock II
class Solution:
     def maxProfit(self, prices):
        profit = 0
        for i in range(len(prices) - 1):
            profit += max(0, prices[i + 1] - prices[i])
        return profit

#------------------------------------------------------------------------------

# Q123 Best Time to Buy and Sell Stock III
class Solution:
    #list.sort() has to be on its own. You can't combine it with other functions
    #reversed() creates a generator. If you don't want that, use list[::-1]
    def maxProfit(self, prices):
        result = []
        i,j = 0,1
        while i < len(prices)-1:
            if prices[i] < prices[j]:
                for k in range(j,len(prices)):
                    if prices[k] > prices[i]:
                        result.append([(i,k),prices[k]-prices[i]])
            i+=1;j+=1
        if not result:
            return 0
        #How to sort by second element
        result = sorted(result,key = lambda x: int(x[1]))[::-1]
        for l in range(len(result)-1):
            if result[l][0][0] > result[l+1][0][1]:
                return(result[l][1]+result[l+1][1])
        return result[0][1]

    def maxProfit2(self, prices):
        #how to do negative infinity
        hold1, hold2 = float("-inf"), float("-inf")
        release1, release2 = 0, 0
        for i in prices:
            release2 = max(release2, hold2 + i)
            hold2 = max(hold2, release1 - i)
            release1 = max(release1, hold1 + i)
            hold1 = max(hold1, -i)
        return release2

#------------------------------------------------------------------------------

# Q135 Candy
class Solution:
    def candy(self, ratings):
        result = [1]*len(ratings)
        for i in range(1,len(ratings)):
            if ratings[i] > ratings[i-1]:
                result[i] = result[i-1]+1
            elif ratings[i] < ratings[i-1]:
                index = i
                while index > 0 and result[index-1] == result[index]:
                    result[index-1] += 1
                    index -= 1
        return sum(result)

#------------------------------------------------------------------------------

# Q136 Single Number
class Solution:
    def singleNumber(self, nums):
        #2∗(a+b+c)−(a+a+b+b+c)=c
        return 2 * sum(set(nums)) - sum(nums)

    def singleNumber2(self, nums):
        a = 0
        for i in nums:
            a ^= i
        return a

    def singleNumber3(self,nums):
        return collections.Counter(nums).most_common()[-1][0]

#------------------------------------------------------------------------------

# Q146 LRUCache
class LRUCache:
    #you can't return anything in init method
    def __init__(self, capacity):
        self.capacity = capacity
        #Since python 3.7, Python dict is an ordered dict. It puts the most recently added on the right
        #The popitem()-returns an arbitrary element (key, value) pair from the dictionary
        self.lookup = {}
        self.size = 0

    def __repr__(self):
        output = []
        for k,v in self.lookup.items():
            output.append(v)
        return '{}'.format(output)

    def get(self, key):
        value = self.lookup[key]
        del self.lookup[key]
        self.lookup[key] = value
        return value

    def put(self, key, value):
        if key not in self.lookup:
            if self.size == self.capacity:
                for k,v in self.lookup.items():
                    del self.lookup[k]
                    break
            self.lookup[key] = value
            self.size += 1
        else:
            del self.lookup[key]
            self.lookup[key] = value

#------------------------------------------------------------------------------

# Q162 Find Peak Element
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

#------------------------------------------------------------------------------

# Q200 Number of Islands
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

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

# Q218 The Skyline Problem
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

#------------------------------------------------------------------------------

# Q220 Contains Duplicate II
class Solution:
    def containsNearbyDuplicate(self, nums,k):
        lookup = {}
        for pos,e in enumerate(nums):
            if e in lookup:
                if abs(lookup[e]-pos) <= k:
                    return True
            lookup[e] = pos
        return False

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

# Q271 Encode and Decode Strings
class Codec:

    def encode(self, strs):
        encoded_str = ''
        for s in strs:
            encoded_str +=  '0000000' + str(len(s)) + s
        return encoded_str

    def decode(self, s):
        i = 0
        strs = []
        while i < len(s):
            l = int(s[i+7])
            strs.append(s[i+8:i+8+l])
            i += 8+l
        return strs

#------------------------------------------------------------------------------

# Q273. Integer to English Words
class Solution:
    def numberToWords(self, num):
        if num == 0:
            return "Zero"

        lookup = {0: "Zero", 1:"One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen", 16: "Sixteen", 17: "Seventeen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty", 30: "Thirty", 40: "Forty", 50: "Fifty", 60: "Sixty", 70: "Seventy", 80: "Eighty", 90: "Ninety"}

        unit = ["", "Thousand", "Million", "Billion"]

        res, i = [], 0
        while num:
            cur = num % 1000
            if num % 1000:
                res.append(self.threeDigits(cur, lookup, unit[i]))
            num //= 1000
            i += 1
        return " ".join(res[::-1])

    def threeDigits(self, num, lookup, unit):
        res = []
        if num // 100:
            res = [lookup[num // 100] + " " + "Hundred"]
        if num % 100:
            res.append(self.twoDigits(num % 100, lookup))
        if unit != "":
            res.append(unit)
        return " ".join(res)

    def twoDigits(self, num, lookup):
        if num in lookup:
            return lookup[num]
        return lookup[(num // 10) * 10] + " " + lookup[num % 10]

#------------------------------------------------------------------------------

# Q285 Inorder Successor in BST
class Solution1:
    def inorderSuccessor(self, root, p):
        self.p = p
        self.found = False

        def inorderTraversal(root):
            if root.left:
                inorderTraversal(root.left)
            if root.val == self.p:
                self.found = True
            #since it uses self.found, it acts like a global variable
            if self.found:
                return root.val
            if root.right:
                inorderTraversal(root.right)

        #this method can only be called after its definition. Python is interpreted, so it goes line by line.
        return inorderTraversal(root)

class Solution2:
    def inorderSuccessor(self, root, p):
        # If it has right subtree.
        if p and p.right:
            p = p.right
            while p.left:
                p = p.left
            return p

        # Search from root.
        successor = None
        while root and root != p:
            if root.val > p.val:
                successor = root
                root = root.left
            else:
                root = root.right

        return successor

#------------------------------------------------------------------------------

# Q303 Range Sum Query - Immutable
from itertools import accumulate

class NumArray:

    def __init__(self, nums):
        self.nums = list(accumulate([0]+nums))

    def sumRange(self, i, j):
        return self.nums[j+1] - self.nums[i]

#------------------------------------------------------------------------------

# Q325 Maximum Size Subarray Sum Equals k
class Solution1:
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

from itertools import accumulate

class Solution2:
    def maxSubArrayLen(self, nums, k):
        lookup = {}
        length = 0
        nums = list(accumulate([0] + nums))
        for pos,e in enumerate(nums):
            if (e - k) in lookup:
                length = max(length,pos - lookup[(e - k)])
            else:
                if e not in lookup:
                    lookup[e] = pos
        return length

#------------------------------------------------------------------------------

# Q393 UTF-8 Validation
class Solution:
    def validUtf8(self, data):
        count = 0
        for c in data:
            if count == 0:
                if (c >> 5) == 0b110:
                    count = 1
                elif (c >> 4) == 0b1110:
                    count = 2
                elif (c >> 3) == 0b11110:
                    count = 3
                elif (c >> 7):
                    return False
            else:
                if (c >> 6) != 0b10:
                    return False
                count -= 1
        return count == 0

#------------------------------------------------------------------------------

# Q539 Minimum Time Difference
class Solution:
    def findMinDifference(self, timePoints):
        minutes = [int(x[:2]) * 60 + int(x[3:]) for x in timePoints]
        minutes.sort()
        return min((y - x) % (24 * 60) for x, y in zip(minutes, minutes[1:] + minutes[:1]))
