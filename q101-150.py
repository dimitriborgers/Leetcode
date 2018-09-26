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


# Q112 Path Sum


# Q113 Path Sum II


# Q114 Flatten Binary Tree to Linked List


# Q115 Distinct Subsequences


# Q116 Populating Next Right Pointers in Each Node


# Q117 Populating Next Right Pointers in Each Node II


# Q118 Pascal's Triangle


# Q119 Pascal's Triangle II


# Q120 Triangle


# Q121


# Q122


# Q123


# Q124


# Q125


# Q126


# Q127


# Q128


# Q129


# Q130


# Q131


# Q132


# Q133


# Q134


# Q135


# Q136


# Q137


# Q138


# Q139


# Q140


# Q141


# Q142


# Q143


# Q144


# Q145


# Q146


# Q147


# Q148


# Q149


# Q150
