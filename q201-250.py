# Q201 Bitwise AND of Numbers Range


# Q202 Happy Number


# Q203 Remove Linked List Elements


# Q204 Count Primes


# Q205 Isomorphic Strings
import collections

class Solution:
    def isIsomorphic(self, s, t):
        seqs = collections.defaultdict(set)
        seqt = collections.defaultdict(set)
        for i in range(len(s)):
            seqs[s[i]].add(t[i])
            seqt[t[i]].add(s[i])
            if len(seqs[s[i]]) > 1 or len(seqt[t[i]]) > 1:
                return False
        return True

# Q206 Reverse Linked List


# Q207 Course Schedule
# DFS - very slow
import collections

class Solution1:
    def canFinish(self, numCourses, prerequisites):
        def dfs(vertex):
            if vertex in lookup:
                #if you don't do above if statement, then lookup will add any vertex that it finds in lookup[vertex] that was not already part of lookup. So the dictionary changes size as you loop.
                for adjacent in lookup[vertex]:
                    if adjacent == target:
                        return True
                    if adjacent not in visited:
                        visited.add(adjacent)
                        tmp = dfs(adjacent)
                        if tmp:
                            return tmp

            return False

        lookup = collections.defaultdict(list)
        for edge in prerequisites:
            lookup[edge[0]].append(edge[1])

        for vertex in lookup:
            target = vertex
            visited = {vertex}
            if dfs(vertex):
                return False

        return True

# Topological Sort
class Solution:
    def canFinish(self, n, edges):
        # construct graph
        graph = {i: set() for i in range(n)}
        in_degrees = {i:0 for i in range(n)}

        for edge in edges:
            graph[edge[0]].add(edge[1])
            in_degrees[edge[1]] += 1

        # init var
        q = collections.deque()
        visited = set()

        # find nodes whose in degree == 0
        for index, in_degree in in_degrees.items():
            if in_degree == 0:
                q.append(index)

        # loop all nodes whose in degree == 0
        while q:
            index = q.popleft()
            visited.add(index)
            for g in graph[index]:
                in_degrees[g] -= 1
                if in_degrees[g] == 0:
                    q.append(g)
        return len(visited) == n

# Q208 Implement Trie (Prefix Tree)
class Trie:

    def __init__(self):
        self.trie = {}

    def insert(self, word):
        cur = self.trie
        for letter in word:
            if letter not in cur:
                cur[letter] = {}
            cur = cur[letter]
        cur['$'] = None

    def search(self, word):
        cur = self.trie
        for letter in word:
            if letter not in cur:
                return False
            cur = cur[letter]
        if '$' in cur:
            return True
        return False

    def startsWith(self, prefix):
        cur = self.trie
        for letter in prefix:
            if letter not in cur:
                return False
            cur = cur[letter]
        return True

# Q209 Minimum Size Subarray Sum


# Q210 Course Schedule II


# Q211 Add and Search Word - Data structure design


# Q212 Word Search II
class TrieNode():
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.isWord = False

class Trie():
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for w in word:
            node = node.children[w]
        node.isWord = True

    def search(self, word):
        node = self.root
        for w in word:
            node = node.children.get(w)
            if not node:
                return False
        return node.isWord

class Solution:
    def findWords(self, board, words):
        res = []
        trie = Trie()
        node = trie.root
        for w in words:
            trie.insert(w)
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(board, node, i, j, "", res)
        return res

    def dfs(self, board, node, i, j, path, res):
        if node.isWord:
            res.append(path)
            node.isWord = False
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
            return
        tmp = board[i][j]
        node = node.children.get(tmp)
        if not node:
            return
        board[i][j] = "#"
        self.dfs(board, node, i+1, j, path+tmp, res)
        self.dfs(board, node, i-1, j, path+tmp, res)
        self.dfs(board, node, i, j-1, path+tmp, res)
        self.dfs(board, node, i, j+1, path+tmp, res)
        board[i][j] = tmp

# Q213 House Robber II
class Solution:
    def rob(self, nums):
        n = len(nums)
        if n == 0:
            return 0
        if n <= 2:
            return max(nums)

        dp = [[0,0] for _ in range(n)]
        dp[0] = [0, nums[0]]
        dp[1] = [nums[1], max(nums[1], nums[0])]

        for i in range(2,n-1):
            dp[i][0] = max(dp[i-1][0],dp[i-2][0] + nums[i])
            dp[i][1] = max(dp[i-1][1],dp[i-2][1] + nums[i])
        lastv = max(dp[n-2][0],dp[n-3][0] + nums[n-1])
        return max(lastv, dp[n-2][1])

# Q214 Shortest Palindrome


# Q215 Kth Largest Element in an Array
# nlgn time
class Solution1:
    def findKthLargest(self, nums, k):
        return sorted(nums)[::-1][k-1]

# n time
class Solution2:
    def findKthLargest(self, nums, k):
        # convert the kth largest to smallest
        return self.findKthSmallest(nums, len(nums)+1-k)

    def findKthSmallest(self, nums, k):
        if nums:
            pos = self.partition(nums, 0, len(nums)-1)
            if k > pos+1:
                return self.findKthSmallest(nums[pos+1:], k-pos-1)
            elif k < pos+1:
                return self.findKthSmallest(nums[:pos], k)
            else:
                return nums[pos]

    def partition(self, nums, l, r):
        low = l
        while l < r:
            if nums[l] < nums[r]:
                nums[l], nums[low] = nums[low], nums[l]
                low += 1
            l += 1
        nums[low], nums[r] = nums[r], nums[low]
        return low

# n time
import heapq

class Solution:
    def findKthLargest(self, nums, k):
        heapq.heapify(nums)

        for _ in range(len(nums)-k):
            heapq.heappop(nums)

        return nums[0]

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
class Solution:
    def maximalSquare(self, matrix):
        dp, maxArea = [[0]*len(matrix[0]) for _ in range(len(matrix))], 0
        for r,row in enumerate(matrix):
            for c,col in enumerate(row):
                if r == 0 or c == 0:
                    dp[r][c] = int(col)
                elif int(col) == 1:
                    dp[r][c] = min(dp[r - 1][c - 1], dp[r][c - 1], dp[r - 1][c]) + 1
                maxArea = max(maxArea, dp[r][c])
        return maxArea*maxArea

# Q222 Count Complete Tree Nodes
class Solution:
    #complete binary tree has 2^depth - 1 nodes
    #in this problem, you don't include the (- 1) because you're always looking at subtrees, which does not include root node
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
class Solution:
    def summaryRanges(self, nums):
        if not nums:
            return []
        end = start = nums[0]
        lst = []
        for item in nums[1:]:
            if item - end != 1:
                if start == end:
                    lst.append(str(start))
                else:
                    lst.append("{}->{}".format(start,end))
                start = item
            end = item
        if start == end:
            lst.append(str(start))
        else:
            lst.append("{}->{}".format(start,end))

        return lst

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
class Solution:
    def productExceptSelf(self, nums):
        p = 1
        n = len(nums)
        output = []
        for i in range(0,n):
            output.append(p)
            p = p * nums[i]
        p = 1
        for i in range(n-1,-1,-1):
            output[i] = output[i] * p
            p = p * nums[i]
        return output

# Q239 Sliding Window Maximum
class Solution:
    def maxSlidingWindow(self, nums, k):
        queue = collections.deque()
        result = []
        for pos,e in enumerate(nums):
            while queue and nums[queue[-1]] < e:
                queue.pop()
            queue += pos,
            if queue[0] == pos - k:
                queue.popleft()
            if pos >= k - 1:
                #this same as result.append(nums[queue[0]])
                result += nums[queue[0]],
        return result

# Q240 Search a 2D Matrix II
class Solution:
    def searchMatrix(self, matrix, target):
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False

        height = len(matrix)
        width = len(matrix[0])

        row = height-1
        col = 0

        while col < width and row >= 0:
            if matrix[row][col] > target:
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            else:
                return True

        return False

# Q241 Different Ways to Add Parentheses


# Q242 Valid Anagram


# Q243 Shortest Word Distance


# Q244 Shortest Word Distance II


# Q245 Shortest Word Distance III


# Q246 Strobogrammatic Number
class Solution:
    def isStrobogrammatic(self, num):
        dic = {'0':'0','1':'1','8':'8','6':'9','9':'6'}

        def check(num):
            if len(num) == 1:
                return True if num[0] in {'0','1','8'} else False
            elif len(num) == 2:
                if num[0] in dic and num[-1] in dic and dic[num[0]] == num[-1] and dic[num[-1]] == num[0]:
                    return True
                return False
            else:
                return num[0] in dic and num[-1] in dic and dic[num[0]] == num[-1] and dic[num[-1]] == num[0] and check(num[1:-1])

        return check(num)

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
    def strobogrammaticInRange(self, low, high):
        a = self.below(high)
        b = self.below(low, include=False)
        return a-b if a>b else 0

    def below(self,n,include=True):
        res = 0
        for i in range(len(n)):
            res += self.number(i)
        l = self.strobogrammatic(len(n))

        if include:
            l = [num for num in l if (len(num)==1 or num[0]!='0') and num <= n]
        else:
            l = [num for num in l if (len(num)==1 or num[0]!='0') and num < n]
        return res + len(l)

    def strobogrammatic(self,l):
        res = []
        if l == 1:
            return ['0','1','8']
        if l == 2:
            return ['00','11','69','96','88']
        for s in self.strobogrammatic(l-2):
            res.append('0'+s+'0')
            res.append('1'+s+'1')
            res.append('6'+s+'9')
            res.append('8'+s+'8')
            res.append('9'+s+'6')
        return res

    def number(self,l):
        if l == 0:
            return 0
        if l%2 == 0:
            return 4*(5**(l//2-1))
        elif l == 1:
            return 3
        else:
            return 3*(5**(l//2-1))*4

# Q249 Group Shifted Strings


# Q250 Count Univalue Subtrees

