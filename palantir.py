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

# Q122 Best Time to Buy and Sell Stock II


#------------------------------------------------------------------------------

# Q123 Best Time to Buy and Sell Stock III


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
