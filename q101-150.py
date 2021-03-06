# Q101 Symmetric Tree
class Solution:
    def isSymmetric(self, root):
        if root is None:
            return True
        stack = []
        stack.append(root.left)
        stack.append(root.right)

        while stack:
            p, q = stack.pop(), stack.pop()

            if p is None and q is None:
                continue

            if p is None or q is None or p.val != q.val:
                return False

            stack.append(p.left)
            stack.append(q.right)

            stack.append(p.right)
            stack.append(q.left)

        return True

# Recursive solution
class Solution2:
    def isSymmetric(self, root):
        if root is None:
            return True

        return self.isSymmetricRecu(root.left, root.right)

    def isSymmetricRecu(self, left, right):
        if left is None and right is None:
            return True
        if left is None or right is None or left.val != right.val:
            return False
        return self.isSymmetricRecu(left.left, right.right) and self.isSymmetricRecu(left.right, right.left)

# Q102 Binary Tree Level Order Traversal
import collections

class Solution:
    def levelOrder(self, root):
        outcome = collections.defaultdict(list)
        self._helper(root,outcome)
        return [[i for i in v] for k,v in outcome.items()]

    def _helper(self,root,dictionary,level=0):
        if root:
            dictionary[level].append(root.val)
        else:
            return
        self._helper(root.left,dictionary,level+1)
        self._helper(root.right,dictionary,level+1)

# Q103 Binary Tree Zigzag Level Order Traversal
import collections

class Solution:
    def zigzagLevelOrder(self, root):
        outcome = collections.defaultdict(list)
        self._helper(root,outcome)
        outcome = [[i for i in v] for k,v in outcome.items()]
        for i in range(len(outcome)):
            if i % 2 == 1:
                #using reversed(i) creates a generator
                outcome[i] = outcome[i][::-1]
        return outcome

    def _helper(self,root,dictionary,level=0):
        if root:
            dictionary[level].append(root.val)
        else:
            return
        self._helper(root.left,dictionary,level+1)
        self._helper(root.right,dictionary,level+1)

# Q104 Maximum Depth of Binary Tree
class Solution:
    def maxDepth(self, root):
        return self._helper(root)

    def _helper(self,root,level=1):
        if not root:
            return level - 1

        return max(self._helper(root.left,level+1),self._helper(root.right,level+1))

# Q105 Construct Binary Tree from Preorder and Inorder Traversal
from collections import deque
class Solution:
    def buildTree(self, preorder, inorder):
        def helper(preorder, inorder):
            if not inorder:
                return None

            root_val = preorder.popleft()
            root = TreeNode(root_val)

            index = inorder.index(root_val)

            root.left= helper(preorder, inorder[:index])
            root.right = helper(preorder, inorder[index + 1:])
            return root

        return helper(deque(preorder), inorder)

# Q106 Construct Binary Tree from Inorder and Postorder Traversal
class Solution:
    def buildTree(self, inorder, postorder):
        return self.buildTreeRecu(postorder, inorder, len(postorder), 0, len(inorder))

    def buildTreeRecu(self, postorder, inorder, post_end, in_start, in_end):
        if in_start == in_end:
            return None
        node = TreeNode(postorder[post_end - 1])
        i = inorder.index(postorder[post_end - 1])
        node.left = self.buildTreeRecu(postorder, inorder, post_end - 1 - (in_end - i - 1), in_start, i)
        node.right = self.buildTreeRecu(postorder, inorder, post_end - 1, i + 1, in_end)
        return node

# Q107 Binary Tree Level Order Traversal II
import collections

class Solution:
    def levelOrderBottom(self, root):
        outcome = collections.defaultdict(list)
        self._helper(root,outcome)
        return [[i for i in v] for k,v in outcome.items()][::-1]

    def _helper(self,root,dictionary,level=0):
        if root:
            dictionary[level].append(root.val)
        else:
            return
        self._helper(root.left,dictionary,level+1)
        self._helper(root.right,dictionary,level+1)

# Q108 Convert Sorted Array to Binary Search Tree
class Solution:
    def sortedArrayToBST(self, nums):
        if not nums:
            return
        head = TreeNode(nums[len(nums)//2])
        head.left = self.sortedArrayToBST(nums[:len(nums)//2])
        head.right = self.sortedArrayToBST(nums[len(nums)//2+1:])
        return head

# Q109 Convert Sorted List to Binary Search Tree
class Solution:
    def sortedListToBST(self, head):
        current, length = head, 0
        while current:
            current, length = current.next, length + 1
        #With self.head, you can use it in any definition of the class. Not necessary to use globals
        self.head = head
        return self.sortedListToBSTRecu(0, length)

    def sortedListToBSTRecu(self, start, end):
        if start == end:
            return None
        mid = start + (end - start) // 2
        left = self.sortedListToBSTRecu(start, mid)
        current = TreeNode(self.head.val)
        current.left = left
        self.head = self.head.next
        current.right = self.sortedListToBSTRecu(mid + 1, end)
        return current

# Q110 Balanced Binary Tree
class Solution:
    def isBalanced(self, root):
        #global int variables can be used in other functions but scoping rules still apply. Once out of scope, they lose value given to them in funciton.
        def getHeight(root):
            if not root:
                return 0

            left, right = getHeight(root.left), getHeight(root.right)

            if left < 0 or right < 0 or abs(left - right) > 1:
                return -1
            return max(left, right) + 1

        return getHeight(root) >= 0

# Q111 Minimum Depth of Binary Tree
class Solution:
    def minDepth(self, root):
        if not root.left and not root.right:
            return 0

        if root.left and not root.right:
            return self.minDepth(root.left)+1
        elif root.right and not root.left:
            return self.minDepth(root.right)+1
        else:
            return min(self.minDepth(root.left)+1,self.minDepth(root.right)+1)

# Q112 Path Sum
class Solution:
    def hasPathSum(self, root: 'TreeNode', sum: 'int') -> 'bool':
        def helper(node,value):
            if value == sum and (not node.left and not node.right):
                return True

            if node.left:
                value += node.left.val
                if helper(node.left,value):
                    return True
                value -= node.left.val

            if node.right:
                value += node.right.val
                if helper(node.right,value):
                    return True
            return False

        if not root:
            return False
        return helper(root,root.val)

# Q113 Path Sum II
class Solution:
    def pathSum(self, root: 'TreeNode', sum: 'int') -> 'bool':
        output = []
        if not root:
            return []

        def helper(node,value,seq):
            if value == sum and (not node.left and not node.right):
                output.append(seq[:])

            if node.left:
                value += node.left.val
                seq.append(node.left.val)
                helper(node.left,value,seq)
                value -= node.left.val
                seq.pop()

            if node.right:
                value += node.right.val
                seq.append(node.right.val)
                helper(node.right,value,seq)
                value -= node.right.val
                seq.pop()
            return False

        helper(root,root.val,[root.val])
        return output

# Q114 Flatten Binary Tree to Linked List
class Solution:
    #can either do self.list_head in __init__ or within normal method (normal method allows you to use it in sub-methods)
    #or simply list_head in class. However, you still need to refer to it as self.list_head when you want to use it in method
    list_head = None
    def flatten(self, root):
        if root != None:
            self.flatten(root.right)
            self.flatten(root.left)
            root.right = self.list_head
            root.left = None
            self.list_head = root

# Q115 Distinct Subsequences
class Solution:
    def numDistinct(self, s, t):
        l1, l2 = len(s)+1, len(t)+1
        dp = [[0] * l1 for _ in range(l2)]

        for j in range(l1):
            dp[0][j] = 1
        for i in range(1, l2):
            for j in range(1, l1):
                if s[j-1] == t[i-1]:
                    dp[i][j] = dp[i][j-1] + min(dp[i-1][j-1],dp[i-1][j])
                else:
                    dp[i][j] = dp[i][j-1]

        return dp[-1][-1]

# Q116 Populating Next Right Pointers in Each Node
class Solution:
    def connect(self, root):
        if not root:
            return root
        level = collections.deque([root])
        while level:
            next_level = collections.deque([])
            for _ in range(len(level)):
                tmp = level.popleft()
                if not tmp:
                    continue
                if len(level) == 0:
                    tmp.next = None
                else:
                    tmp.next = level[0]
                next_level += tmp.left,tmp.right
            level = next_level

# Q117 Populating Next Right Pointers in Each Node II
class Solution:
    def connect(self, root):
        if not root:
            return root
        level = collections.deque([root])
        while level:
            next_level = collections.deque([])
            for _ in range(len(level)):
                tmp = level.popleft()
                if len(level) == 0:
                    tmp.next = None
                else:
                    tmp.next = level[0]
                if tmp.left:
                    next_level += tmp.left,
                if tmp.right:
                    next_level += tmp.right,
            level = next_level

# Q118 Pascal's Triangle
class Solution:
    result = []
    def generate(self, numRows):
        if numRows == 0:
            return []
        if numRows == 1:
            return [[1]]
        elif numRows == 2:
            return [[1],[1,1]]
        else:
            self.result = self.generate(2)
            for i in range(2,numRows):
                seq = [1]
                for i in range(len(self.result[-1])-1):
                    seq.append(self.result[-1][i]+self.result[-1][i+1])
                seq.append(1)
                self.result.append(seq)
        return self.result

# Q119 Pascal's Triangle II
class Solution:
    def getRow(self, rowIndex):
        result = [0] * (rowIndex + 1)
        for i in range(rowIndex + 1):
            old = result[0] = 1
            for j in range(1, i + 1):
                old, result[j] = result[j], old + result[j]
        return result

# Q120 Triangle
import functools

class Solution:
    def minimumTotal(self, triangle: 'List[List[int]]') -> 'int':
        if not triangle:
            return 0

        @functools.lru_cache(maxsize = None)
        def dfs(level,total,loc):
            if level < len(triangle)-1:
                return total + min(dfs(level+1,triangle[level+1][loc],loc),dfs(level+1,triangle[level+1][loc+1],loc+1))

            else:
                return triangle[level][loc]

        return dfs(0,triangle[0][0],0)

# Q121 Best Time to Buy and Sell Stock
class Solution:
    def maxProfit(self, prices):
        #how to create infinity
        max_profit, min_price = 0, float("inf")
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        return max_profit

# Q122 Best Time to Buy and Sell Stock II
class Solution:
     def maxProfit(self, prices):
        profit = 0
        for i in range(len(prices) - 1):
            profit += max(0, prices[i + 1] - prices[i])
        return profit

# Q123 Best Time to Buy and Sell Stock III
class Solution:
    def maxProfit(self, prices):
        if not prices:
            return 0

        profits = []
        max_profit = 0
        current_min = prices[0]
        for price in prices:
            current_min = min(current_min, price)
            max_profit = max(max_profit, price - current_min)
            profits.append(max_profit)

        total_max = 0
        max_profit = 0
        current_max = prices[-1]
        for i in range(len(prices) - 1, -1, -1):
            current_max = max(current_max, prices[i])
            max_profit = max(max_profit, current_max - prices[i])
            total_max = max(total_max, max_profit + profits[i])

        return total_max

# Q124 Binary Tree Maximum Path Sum
class Solution:
    def maxPathSum(self, root: 'TreeNode') -> 'int':
        total = [root.val]
        if not root.left and not root.right:
            return root.val

        def recur(node):
            if not node:
                return 0

            child_left = max(recur(node.left),0)
            child_right = max(recur(node.right),0)
            total[0] = max(total[0],child_left + node.val + child_right)

            return node.val + max(child_left,child_right)

        recur(root)
        return total[0]

# Q125 Valid Palindrome
class Solution:
    def isPalindrome(self, s):
        #a.isalnum() returns true if all characters in the string are alphanumeric or numbers and there is at least one character.
        #"this2009".isalnum() would be True
        tmp = ''
        for i in s:
            if i.isalnum():
                tmp += i.lower()
        for i in range(len(tmp)//2):
            if tmp[i] != tmp[~i]:
                return False
        return True

# Q126 Word Ladder II
class Solution:
    def findLadders(self, beginWord, endWord, wordList):
        wordList = set(wordList)
        result = []
        level = {beginWord:[[beginWord]]}

        # bfs using levels
        while level:
            newlevel = collections.defaultdict(list)
            for word in level:
                if word == endWord:
                    result.extend(level[word][:])
                else:
                    for i in range(len(word)):
                        for c in 'abcdefghijklmnopqrstuvwxyz':
                            neww = word[:i]+c+word[i+1:]
                            if neww in wordList:
                                #idea of path creation in dfs
                                newlevel[neww] += [j+[neww] for j in level[word]]

            wordList -= set(newlevel)
            level = newlevel
        return result

# Q127 Word Ladder
class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        distance = 0
        #[beginWord] creates ['hit'], list(beginWord) creates ['h','i','t']
        level = [beginWord]
        visited = set([beginWord])
        lookup = set(wordList)

        while level:
            next_level = []

            for word in level:
                if word == endWord:
                    return distance + 1
                for i in range(len(word)):
                    #Can't do sets() of required letters because you don't know how many sets you would need, which would require making a list of sets
                    for j in 'abcdefghijklmnopqrstuvwxyz':
                        #instead of creating a temporary list of word and changing it's value at an index, just concatenate letters
                        #strs are not considered lists, so you don't have to worry about references
                        candidate = word[:i] + j + word[i + 1:]
                        if candidate not in visited and candidate in lookup:
                            next_level.append(candidate)
                            visited.add(candidate)
            distance += 1
            level = next_level
        return 0

# Q128 Longest Consecutive Sequence
class Solution:
    #(n) since you hashSet
    def longestConsecutive(self, nums):
        longest_streak = 0
        num_set = set(nums)

        for num in num_set:
            #check to make sure you're at the lowest one, which prevents O(n^2)
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1

                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)
        return longest_streak

# Q129 Sum Root to Leaf Numbers
class Solution:
    def sumNumbers(self, root: 'TreeNode') -> 'int':
        if not root:
            return 0

        level = [root]
        total = 0
        while level:
            next_level = []
            #always be careful when looping through list without range(len()). If you modify the list, you may skip elements.
            for _ in range(len(level)):
                num = level.pop()
                if not num.left and not num.right:
                    total += num.val
                if num.left:
                    tmp = num.val * 10
                    num.left.val += (tmp)
                    next_level.append(num.left)
                if num.right:
                    tmp = num.val * 10
                    num.right.val += (tmp)
                    next_level.append(num.right)
            level = next_level
        return total

# Q130 Surrounded Regions
class Solution:
    def solve(self, board):
        queue = collections.deque([])
        for r in range(len(board)):
            for c in range(len(board[0])):
                if (r in [0, len(board)-1] or c in [0, len(board[0])-1]) and board[r][c] == "O":
                    queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            if 0<=r<len(board) and 0<=c<len(board[0]) and board[r][c] == "O":
                board[r][c] = "D"
                queue.append((r-1, c)); queue.append((r+1, c))
                queue.append((r, c-1)); queue.append((r, c+1))

        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == "O":
                    board[r][c] = "X"
                elif board[r][c] == "D":
                    board[r][c] = "O"

# Q131 Palindrome Partitioning
#Can't do a class attribute that is a list, because that list will be modified for all instances of the class, since it is mutable and by reference
class Solution:
    def partition(self, s):
        res = []
        self.dfs(s, [], res)
        return res

    def dfs(self, s, path, res):
        if not s:
            res.append(path)
            return
        for i in range(1, len(s)+1):
            if self.isPal(s[:i]):
                #this creates a new instance of the path since you're adding something to it
                self.dfs(s[i:], path+[s[:i]], res)

    def isPal(self, s):
        return s == s[::-1]

# Q132 Palindrome Partitioning II
class Solution:
    def minCut(self, s):
        def isPal(s):
            return s== s[::-1]
        cnt=0
        queue=[0]
        visited = [0]*len(s)
        while True:
            temp = []
            while queue:
                cur = queue.pop(0)
                for i in range (len(s)-1, cur-1, -1):
                    if not visited[i] and isPal(s[cur:i+1]):
                        if i == len(s)-1:
                            return cnt
                        else:
                            temp.append(i+1)
                visited[cur] =1
            cnt+=1
            queue += temp

# Q133 Clone Graph
import collections

class Solution:
    def cloneGraph(self, node):
        if not node:
            return

        #deque(iterable), thus element inside () should have iter() defined.
        queue = collections.deque()
        first_node = UndirectedGraphNode(node.label)
        queue.append(node)
        graph = {node:first_node}

        while queue:
            tmp_ori = queue.popleft()
            tmp_copy = graph[tmp_ori]
            for n in tmp_ori.neighbors:
                if n not in graph:
                    neighbor_node = UndirectedGraphNode(n.label)
                    queue.append(n)
                    tmp_copy.neighbors.append(neighbor_node)
                    graph[n] = neighbor_node
                else:
                    tmp_copy.neighbors.append(graph[n])

        return first_node

# Q134 Gas Station
class Solution:
    def canCompleteCircuit(self, gas, cost):
        length = len(gas)
        for i in range(len(gas)):
            if cost[i] > gas[i]:
                continue
            starting = i
            current = gas[i]-cost[i]
            while current >= 0:
                i+=1
                if i % length == starting:
                    return True
                current += (gas[i%length]-cost[i%length])
        return False

# Q135 Candy
class Solution:
    def candy(self,ratings):
        candies = [1]*len(ratings)
        for i in range(1,len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1;

        total = candies[-1]
        for i in range(len(ratings)-2,-1,-1):
            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)
            total += candies[i]
        return total

# Q136 Single Number
class Solution:
    #To do a union/intersection of lists, you first have to turn them into sets
    def singleNumber(self, nums):
        #2∗(a+b+c)−(a+a+b+b+c)=c
        return 2 * sum(set(nums)) - sum(nums)

    def singleNumber2(self, nums):
        a = 0
        for i in nums:
            #xor
            #2 xor 3 xor 2 -> 2's cancel each other out, only 3 is left
            #this only works if max 2 duplicates to cancel out
            a ^= i
        return a

    def singleNumber3(self,nums):
        return collections.Counter(nums).most_common()[-1][0]

# Q137 Single Number II
class Solution:
    def singleNumber(self, A):
        one, two = 0, 0
        for x in A:
            one, two = (~x & one) | (x & ~one & ~two), (~x & two) | (x & one)
        return one

# Q138 Copy List with Random Pointer
class Solution:
    def copyRandomList(self, head):
        lookup = {}
        cur = head
        new_head = None
        while cur:
            if cur not in lookup:
                new_cur = RandomListNode(cur.label)
                lookup[cur] = new_cur
                if not new_head:
                    new_head = new_cur
            if cur.next:
                if cur.next not in lookup:
                    new_next = RandomListNode(cur.next.label)
                    lookup[cur.next] = new_next
                lookup[cur].next = lookup[cur.next]
            if cur.random:
                if cur.random not in lookup:
                    new_random = RandomListNode(cur.random.label)
                    lookup[cur.random] = new_random
                lookup[cur].random = lookup[cur.random]
            cur = cur.next
        return new_head

# Q139 Word Break
class Solution:
    #Cannot use lru_cache here because one of the arguments is a list, thus mutable
    #lru_cache or memoization in general will only work when all arguments are immutable
    def wordBreak(self, s, wordDict):
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for i in range(len(s)):
            for j in range(i, len(s)):
                if dp[i] and s[i: j+1] in wordDict:
                    dp[j+1] = True

        return dp[-1]

# Q140 Word Break II
class Solution:
    def wordBreak(self, s, wordDict):
        return self.helper(0, s, set(wordDict), {})

    def helper(self, k, s, wordDict, cache):
        if k == len(s):
            return []
        elif k in cache:
            return cache[k]
        else:
            cache[k] = []
            for i in range(k, len(s)):
                left = s[k:i+1]
                if left in wordDict:
                    remainder = self.helper(i+1, s, wordDict, cache)
                    if remainder:
                        for x in remainder:
                            cache[k].append(left + " " + x)
                    elif (i == len(s)-1):
                        cache[k].append(left)
            return cache[k]

# Q141 Linked List Cycle
class Solution:
    def hasCycle(self, head):
        #not considered part of space complexity because heap objects are already there
        fast, slow = head, head
        while fast and fast.next:
            fast, slow = fast.next.next, slow.next
            #use is instead of == because you want to make sure it's the same instance of it as well, not just a copy
            if fast is slow:
                return True
        return False

# Q142 Linked List Cycle II
class Solution:
    def detectCycle(self, head):
        visited = set()

        node = head
        while node is not None:
            if node in visited:
                return node
            else:
                visited.add(node)
                node = node.next
        return None

# Q143 Reorder List
class Solution:
    def reorderList(self, head):
        if not head or not head.next or not head.next.next:
            return
        slow, fast=head, head
        while fast.next and fast.next.next:
            slow, fast=slow.next, fast.next.next
        head1, head2=head, slow.next
        slow.next, cur, pre=None, head2, None
        while cur:
            curnext=cur.next
            cur.next=pre
            pre=cur
            cur=curnext
        cur1, cur2=head1, pre
        while cur2:
            next1, next2=cur1.next, cur2.next
            cur1.next=cur2
            cur2.next=next1
            cur1, cur2=next1, next2

# Q144 Binary Tree Preorder Traversal
class Solution:
    def preorderTraversal(self, root: 'TreeNode') -> 'List[int]':
        output = []

        def recur(node):
            if not node:
                return
            output.append(node.val)
            recur(node.left)
            recur(node.right)

        recur(root)
        return output

# Q145 Binary Tree Postorder Traversal
class Solution:
    def postorderTraversal(self, root: 'TreeNode') -> 'List[int]':
        output = []

        def recur(node):
            if not node:
                return
            recur(node.left)
            recur(node.right)
            output.append(node.val)

        recur(root)
        return output

# Q146 LRU Cache
"""
Cache algorithms are optimizing instructions that a computer program can utilize in order to manage a cache of information stored on the computer. Caching improves performance by keeping recent or often-used data items in a memory locations that are faster or computationally cheaper to access than normal memory stores.
LRU-General implementations of this technique require keeping "age bits" for cache-lines and track the "Least Recently Used" cache-line based on age-bits.
MRU-When a file is being repeatedly scanned in a [Looping Sequential] reference pattern, MRU is the best replacement algorithm.
RR-Randomly selects a candidate item and discards it to make space when necessary. This algorithm does not require keeping any information about the access history.
LFU-Counts how often an item is needed. Those that are used least often are discarded first.
"""
#This solution only works for 3.7+
class LRUCache:
    #you can't return anything in init method
    def __init__(self, capacity):
        self.capacity = capacity
        #Since python 3.7, Python dict is an ordered dict. It puts the most recently added on the right
        #The popitem()-returns an arbitrary element (key, value) pair from the dictionary. No longer the case in 3.7 (it returns last element added)
        self.lookup = {}
        self.size = 0

    def __repr__(self):
        output = []
        for k,v in self.lookup.items():
            output.append(v)
        return '{}'.format(output)

    def get(self, key):
        if key not in self.lookup:
            return -1
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

#This solution works for any Python 3 version
class Node:
    def __init__(self, k, v):
        self.key = k
        self.val = v
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.dict = {}
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        if key in self.dict:
            n = self.dict[key]
            self._remove(n)
            self._add(n)
            return n.val
        return -1

    def put(self, key, value):
        if key in self.dict:
            self._remove(self.dict[key])
        n = Node(key, value)
        self._add(n)
        self.dict[key] = n
        if len(self.dict) > self.capacity:
            n = self.head.next
            self._remove(n)
            del self.dict[n.key]

    def _remove(self, node):
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p

    def _add(self, node):
        p = self.tail.prev
        p.next = node
        self.tail.prev = node
        node.prev = p
        node.next = self.tail

# Q147 Insertion Sort List
class Solution:
    def insertionSortList(self, head):
        p = dummy = ListNode(0)
        cur = dummy.next = head
        while cur and cur.next:
            val = cur.next.val
            if cur.val < val:
                cur = cur.next
                continue
            if p.next.val > val:
                p = dummy
            while p.next.val < val:
                p = p.next
            new = cur.next
            cur.next = new.next
            new.next = p.next
            p.next = new
        return dummy.next

# Q148 Sort List
# easier to do mergelists than quicksort with linked lists
class Solution:
    def sortList(self, head):
        if not head or not head.next:
            return head

        fast, slow, prev = head, head, None
        #to reach halfway point, just have one pointer go twice as fast as the previous one
        while fast and fast.next:
            prev, fast, slow = slow, fast.next.next, slow.next
        #cut list in half
        prev.next = None

        sorted_l1 = self.sortList(head)
        sorted_l2 = self.sortList(slow)

        return self.mergeTwoLists(sorted_l1, sorted_l2)

    def mergeTwoLists(self, l1, l2):
        dummy = ListNode(-1)

        #build up from this dummy
        cur = dummy
        while l1 and l2:
            #since you only ever return the head of a list, you can access the value
            if l1.val <= l2.val:
                cur.next, cur, l1 = l1, l1, l1.next
            else:
                cur.next, cur, l2 = l2, l2, l2.next
        if l1:
            cur.next = l1
        if l2:
            cur.next = l2
        return dummy.next

# Q149 Max Points on a Line
class Solution:
    def maxPoints(self, points):
        def max_points_on_a_line_containing_point_i(i):
            def add_line(i, j, count, duplicates):

                x1 = points[i].x
                y1 = points[i].y
                x2 = points[j].x
                y2 = points[j].y

                if x1 == x2 and y1 == y2:
                    duplicates += 1

                elif y1 == y2:
                    nonlocal horisontal_lines
                    horisontal_lines += 1
                    count = max(horisontal_lines, count)
                else:
                    slope = (x1 - x2) / (y1 - y2)
                    lines[slope] = lines.get(slope, 1) + 1
                    count = max(lines[slope], count)
                return count, duplicates

            lines, horisontal_lines = {}, 1
            count = 1
            duplicates = 0
            for j in range(i + 1, n):
                count, duplicates = add_line(i, j, count, duplicates)
            return count + duplicates

        n = len(points)
        if n < 3:
            return n

        max_count = 1
        for i in range(n - 1):
            max_count = max(max_points_on_a_line_containing_point_i(i), max_count)
        return max_count

# Q150 Evaluate Reverse Polish Notation
class Solution:
    def evalRPN(self, tokens):
        stack = []
        for t in tokens:
            if t not in ["+", "-", "*", "/"]:
                stack.append(int(t))
            else:
                r, l = stack.pop(), stack.pop()
                if t == "+":
                    stack.append(l+r)
                elif t == "-":
                    stack.append(l-r)
                elif t == "*":
                    stack.append(l*r)
                else:
                    # print(6//-12) or print(-6/12) -> -1 instead of 0
                    if l*r < 0 and l % r != 0:
                        stack.append(l//r+1)
                    else:
                        stack.append(l//r)
        return stack.pop()
