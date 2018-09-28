# Q101 Symmetric Tree
class Solution:
    def isSymmetric(self,root):
        return self.inorder(root.left) == self.inorder(root.right)

    def inorder(self,root):
        global result
        result = []
        temp = self._inorder_helper(root)
        return temp

    def _inorder_helper(self,root):
        if root.left:
            self._inorder_helper(root.left)
        result.append(root.val)
        if root.right:
            self._inorder_helper(root.right)
        return result

class Solution1:
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
    def levelOrder(self, root):
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
class Solution:
    def buildTree(self, preorder, inorder):
        return self.buildTreeRecu(preorder, inorder, 0, 0, len(inorder))

    def buildTreeRecu(self, preorder, inorder, pre_start, in_start, in_end):
        if in_start == in_end:
            return None
        node = TreeNode(preorder[pre_start])
        i = inorder.index(preorder[pre_start])
        node.left = self.buildTreeRecu(preorder, inorder, pre_start + 1, in_start, i)
        node.right = self.buildTreeRecu(preorder, inorder, pre_start + 1 + i - in_start, i + 1, in_end)
        return node

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
            return 1

        if root.left and not root.right:
            return self.minDepth(root.left)+1
        elif root.right and not root.left:
            return self.minDepth(root.right)+1
        else:
            return min(self.minDepth(root.left)+1,self.minDepth(root.right)+1)

# Q112 Path Sum
class Solution:
    def hasPathSum(self, root, sum_):

        def helper(root,sum_,current):
            if not root:
                return False
            current += root.val
            if current == sum_ and not root.left and not root.right:
                return True
            return helper(root.left,sum_,current) or helper(root.right,sum_,current)

        return helper(root,sum_,0)

# Q113 Path Sum II
class Solution:
    def hasPathSum(self, root, sum_):
        #instead of global, you can just do a self.something
        #if you write this self.something outside of method, but in class, you have to include it in a def __init__ wrapper
        self.list = []

        def helper(root,sum_,current,seq=None):
            if not seq:
                seq = []
            if not root:
                return
            current += root.val
            seq.append(root.val)
            if current == sum_ and not root.left and not root.right:
                self.list.append(seq)
            return helper(root.left,sum_,current,list(seq)) or helper(root.right,sum_,current,list(seq))

        helper(root,sum_,0)
        return self.list

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
            return root

# Q115 Distinct Subsequences
class Solution:
    stack = []
    result = 0
    j = 0
    def numDistinct(self, s, t):
        if self.j == len(t):
            return self.result

        for i in range(len(s)):
            if s[i] == t[self.j]:
                self.stack.append(s[i])
                self.j += 1
                self.numDistinct(s[i+1:],t)
                if len(self.stack) == len(t):
                    self.result += 1
                    self.stack.pop()
                    self.j -= 1
            else:
                continue
        if self.stack:
            self.stack.pop()
            self.j -= 1
            return self.result
        return self.result

# Q116 Populating Next Right Pointers in Each Node
class TreeLinkNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None

    def __repr__(self):
        return "{} -> {}".format(self.val, repr(self.next))

class Solution:
    def connect(self, root):
        if root.left:
            if root.left.left:
                leftleft = root.left.left
                leftright = root.left.right
                rightleft = root.right.left
                rightright = root.right.right
                self.connector(leftleft,leftright)
                self.connector(leftright,rightleft)
                self.connector(rightleft,rightright)
            self.connector(root.left,root.right)
            self.connect(root.left)
            self.connect(root.right)
        return root

    def connector(self,left,right):
        left.next = right

# Q117 Populating Next Right Pointers in Each Node II
class Solution:
    def connect(self, root):
        leftleft = leftright = rightleft = rightright = None
        if root.left and root.right:
            if root.left.left:
                leftleft = root.left.left
            if root.left.right:
                leftright = root.left.right
            if root.right.left:
                rightleft = root.right.left
            if root.right.right:
                rightright = root.right.right

            if leftleft and leftright:
                self.connector(leftleft,leftright)
            if leftright and rightleft:
                self.connector(leftright,rightleft)
            if rightleft and rightright:
                self.connector(rightleft,rightright)
            if leftright and not rightleft and rightright:
                self.connector(leftright,rightright)
            if leftleft and not leftright and rightleft:
                self.connector(leftleft,rightleft)
            if leftleft and not leftright and not rightleft and rightright:
                self.connector(leftleft,rightright)

            self.connector(root.left,root.right)
            self.connect(root.left)
            self.connect(root.right)
        return root

    def connector(self,left,right):
        left.next = right

class Solution2:
    def connect(self, root):
        head = root
        while head:
            prev, cur, next_head = None, head, None
            while cur:
                if next_head is None:
                    if cur.left:
                        next_head = cur.left
                    elif cur.right:
                        next_head = cur.right

                if cur.left:
                    if prev:
                        prev.next = cur.left
                    prev = cur.left

                if cur.right:
                    if prev:
                        prev.next = cur.right
                    prev = cur.right

                cur = cur.next
            head = next_head

# Q118 Pascal's Triangle
class Solution:
    result = []
    def generate(self, numRows):
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
class Solution:
    def minimumTotal(self, triangle):
        result = [0]*len(triangle)
        result[0] = triangle[0][0]
        for i in range(1,len(triangle)):
            resultTemp = list(result)
            for j in range(len(triangle[i])-1):
                old = resultTemp[j]
                for k in range(j,j+2):
                    print(result)
                    if result[k] != resultTemp[k]:
                        result[k] = old + triangle[i][k] if old + triangle[i][k] < compare[k] else result[k]
                    else:
                        result[k] = old + triangle[i][k]
                        compare = result
        return result

# Q121
class Solution:
    def maxProfit(self, prices):
        left,right = 0,len(prices)-1
        least,most = prices[left],prices[right]
        while left != right:
            if prices[left] >= prices[right]:
                left += 1
            else:
                right -= 1
            least = prices[left] if prices[left] < least else least
            most = prices[right] if prices[right] > most else most
        return most-least if most>least else None

    def maxProfit2(self, prices):
        #how to create infinity
        max_profit, min_price = 0, float("inf")
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        return max_profit

# Q122
class Solution:
     def maxProfit(self, prices):
        profit = 0
        for i in range(len(prices) - 1):
            profit += max(0, prices[i + 1] - prices[i])
        return profit

# Q123
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

# Q124
class Solution:
    maxSum = float("-inf")

    def maxPathSum(self, root):
        self.maxPathSumRecu(root)
        return self.maxSum

    def maxPathSumRecu(self, root):
        if root is None:
            return 0
        left = max(0, self.maxPathSumRecu(root.left))
        right = max(0, self.maxPathSumRecu(root.right))
        self.maxSum = max(self.maxSum, root.val + left + right)
        return root.val + max(left, right)

# Q125
class Solution:
    def isPalindrome(self, s):
        #a.isalnum() returns true if all characters in the string are alphanumeric or numbers and there is at least one character.
        #"this2009".isalnum() would be True
        i,j = 0,len(s)-1
        while i != j:
            if s[i].isalpha() and s[j].isalpha() and s[i].lower() == s[j].lower():
                i+=1
                j-=1
            elif not s[i].isalpha():
                i+=1
            elif not s[j].isalpha():
                j-=1
            else:
                return False
        return True

# Q126 Word Ladder II
class Solution:
    def findLadders(self, start, end, dic):
        dic.add(start)
        dic.add(end)

        result, cur, visited, found, trace = [], [start], set([start]), False, {word: [] for word in dic}

        while cur and not found:
            for word in cur:
                visited.add(word)

            next = set()
            for word in cur:
                for i in range(len(word)):
                    for j in 'abcdefghijklmnopqrstuvwxyz':
                        candidate = word[:i] + j + word[i + 1:]
                        if candidate not in visited and candidate in dic:
                            if candidate == end:
                                found = True
                            next.add(candidate)
                            trace[candidate].append(word)
            cur = next

        if found:
            self.backtrack(result, trace, [], end)

        return result

    def backtrack(self, result, trace, path, word):
        if not trace[word]:
            result.append([word] + path)
        else:
            for prev in trace[word]:
                self.backtrack(result, trace, [word] + path, prev)

# Q127 Word Ladder
class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        distance = 0
        #[beginWord] creates ['hit'], list(beginWord) creates ['h','i','t']
        cur = [beginWord]
        visited = set([beginWord])
        lookup = set(wordList)

        while cur:
            next_queue = []

            for word in cur:
                if word == endWord:
                    return distance + 1
                for i in range(len(word)):
                    #Can't do sets() of required letters because you don't know how many sets you would need, which would require making a list of sets
                    for j in 'abcdefghijklmnopqrstuvwxyz':
                        #instead of creating a temporary list of word and changing it's value at an index, just concatenate letters
                        #strs are not considered lists, so you don't have to worry about references
                        candidate = word[:i] + j + word[i + 1:]
                        if candidate not in visited and candidate in lookup:
                            next_queue.append(candidate)
                            visited.add(candidate)
            distance += 1
            cur = next_queue
        return 0

beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]
#if you have a __init__ method, just put args in Solution brackets
print(Solution().ladderLength(beginWord,endWord,wordList))

# Q128 Longest Consecutive Sequence
class Solution:
    #(n) since you hashSet
    def longestConsecutive(self, nums):
        longest_streak = 0
        num_set = set(nums)

        for num in num_set:
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
    result = []
    def sumNumbers(self, root):
        if not root:
            return 0
        self.sumNumbersRec(root,0)
        output = 0
        for i in self.result:
            output += i
        return output

    def sumNumbersRec(self,root,cur):
        if not cur:
            cur = root.val
        else:
            cur = cur*10 + root.val

        if root.left and root.right:
            self.sumNumbersRec(root.left,cur)
            self.sumNumbersRec(root.right,cur)
        elif root.left:
            self.sumNumbersRec(root.left,cur)
        elif root.right:
            self.sumNumbersRec(root.right,cur)
        else:
            self.result.append(cur)

# Q130 Surrounded Regions
class Solution:
    def solve(self, board):
        n = len(board)
        m = len(board[0])

        for i in range(1,n-1):
            for j in range(1,m-1):
                if j == 1 and i == 1:
                    if (board[i][j-1] == 'O' or board[i-1][j] == 'O') and board[i][j] == 'O':
                        continue
                    else:
                        board[i][j] = 'X'
                elif j == 1 and i == m - 1:
                    if (board[i][j-1] == 'O' or board[i+1][j] == 'O') and board[i][j] == 'O':
                        continue
                    else:
                        board[i][j] = 'X'
                elif j == m-1 and i == 1:
                    if (board[i][j+1] == 'O' or board[i-1][j] == 'O') and board[i][j] == 'O':
                        continue
                    else:
                        board[i][j] = 'X'
                elif j == m-1 and i == n-1:
                    if (board[i][j+1] == 'O' or board[i+1][j] == 'O') and board[i][j] == 'O':
                        continue
                    else:
                        board[i][j] = 'X'
                elif j == 1:
                    if board[i][j-1] == 'O' and board[i][j] == 'O':
                        continue
                    else:
                        board[i][j] = 'X'
                elif j == m-1:
                    if board[i][j+1] == 'O' and board[i][j] == 'O':
                        continue
                    else:
                        board[i][j] = 'X'
                elif i == 1:
                    if board[i-1][j] == 'O' and board[i][j] == 'O':
                        continue
                    else:
                        board[i][j] = 'X'
                elif i == n-1:
                    if board[i+1][j] == 'O' and board[i][j] == 'O':
                        continue
                    else:
                        board[i][j] = 'X'
                else:
                    board[i][j] = 'X'
        return board

# Q131 Palindrome Partitioning
class Solution:
    total = []
    def partition(self, s):
        self.recur(s)
        return self.total

    def recur(self,s,result=None):
        if len(s) <= 1:
            return
        if not result:
            result = []
        for i in range(len(s)-1):
            left = s[:i+1]
            right = s[i+1:]
            if not right:
                return
            if left == left[::-1]:
                result.append(left)
            else:
                continue
            if right == right[::-1]:
                result.append(right)
                self.total.append(list(result))
                result.pop()
            self.recur(right,result)
            result.pop()
        return

# Q132 Palindrome Partitioning II
class Solution:
    total = []
    def partition(self, s):
        self.recur(s)
        #can't make minimum = float('inf') because len() would not work
        minimum = self.total[0]
        for i in self.total:
           minimum = i if len(i) < len(minimum) else minimum
        return len(minimum)

    def recur(self,s,result=None):
        if len(s) <= 1:
            return
        if not result:
            result = []
        for i in range(len(s)-1):
            left = s[:i+1]
            right = s[i+1:]
            if not right:
                return
            if left == left[::-1]:
                result.append(left)
            else:
                continue
            if right == right[::-1]:
                result.append(right)
                self.total.append(list(result))
                result.pop()
            self.recur(right,result)
            result.pop()
        return

# Q133 Clone Graph
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []

class Solution:
    def cloneGraph(self, node):
        if not node:
            return
        cloned_node = UndirectedGraphNode(node.label)
        cloned, queue = {node:cloned_node}, [node]

        while queue:
            current = queue.pop()
            #accessing list of neighbors
            for neighbor in current.neighbors:
                if neighbor not in cloned:
                    queue.append(neighbor)
                    cloned_neighbor = UndirectedGraphNode(neighbor.label)
                    cloned[neighbor] = cloned_neighbor
                cloned[current].neighbors.append(cloned[neighbor])
        return cloned[node]

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
    def candy(self, ratings):
        result = [0]*len(ratings)
        result[0] = 1
        for i in range(1,len(ratings)):
            print(result)
            if ratings[i] > ratings[i-1]:
                result[i] = result[i-1]+1
            elif ratings[i] < ratings[i-1]:
                result[i-1] += 1
                result[i] = 1
                index = i-2
                while index >= 0:
                    if ratings[index] > ratings[index+1]:
                        result[index] += 1
                        index -= 1
                    #putting a continue here would just go one further in the while loop. Continue is used in both while and for loops.
                    break
            else:
                result[i] = 1
        return result

# Q136 Single Number


# Q137 Single Number II


# Q138 Copy List with Random Pointer


# Q139 Word Break


# Q140 Word Break II


# Q141 Linked List Cycle


# Q142 Linked List Cycle II


# Q143 Reorder List


# Q144 Binary Tree Preorder Traversal


# Q145 Binary Tree Postorder Traversal


# Q146 LRU Cache


# Q147 Insertion Sort List


# Q148 Sort List


# Q149 Max Points on a Line


# Q150 Evaluate Reverse Polish Notation
