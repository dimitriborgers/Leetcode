# Q524 Longest Word in Dictionary through Deleting
class Solution:
    def findLongestWord(self, s, d):
        def searcher(s,word):
            if not word:
                return True
            if word[0] in s:
                return searcher(s[s.find(word[0])+1:],word[1:])
            return False

        result = [(len(word),word) for word in d if searcher(s,word)]
        length = max(result,key=lambda x:x[0])[0] if result else 0
        #return min() since you want the lowest alphabetic result if multiple words have same length
        return min([word for num,word in result if num == length]) if result else ''

# Q527 Word Abbreviation
class Solution:
    def wordsAbbreviation(self, dict):
        abb = collections.defaultdict(int)
        for i, w in enumerate(dict):
            for j in range(1, len(w) - 2):
                abb[w[:j] + str(len(w) - j - 1) + w[-1]] += 1
        for i, w in enumerate(dict):
            for j in range(1, len(w) - 2):
                new = w[:j] + str(len(w) - j - 1) + w[-1]
                if abb[new] == 1:
                    dict[i] = new
                    break
        return dict

# Q528 Random Pick with Weight
#Space: O(1) Time: O(n)
import random,bisect,itertools

class Solution:

    def __init__(self, w):
        self._prefix_sum = list(itertools.accumulate(w))

    def pickIndex(self):
        target = random.randint(0, self._prefix_sum[-1]-1)
        #gives the index
        #bisect == bisect_right: if you have [1,1,1], index will be 3
        #bisect_left: if you have [1,1,1], index will be 0. So you actually take the index of the first instance found
        #if element you're inputting isn't in list already, then both methods should return the same index
        return bisect.bisect_right(self._prefix_sum, target)

# Q535 Encode and Decode TinyURL
class Codec:

    alphabet = string.ascii_letters + '0123456789'

    def __init__(self):
        self.url2code = {}
        self.code2url = {}

    def encode(self, longUrl):
        while longUrl not in self.url2code:
            #You can access static variables using self or className, but using self makes it ambiguous as to what type of variable you are accessing. So, better to use className.staticVariable
            code = ''.join(random.choice(Codec.alphabet) for _ in range(6))
            if code not in self.code2url:
                self.code2url[code] = longUrl
                self.url2code[longUrl] = code
        return 'http://tinyurl.com/' + self.url2code[longUrl]

    def decode(self, shortUrl):
        return self.code2url[shortUrl[-6:]]

# Q539 Minimum Time Difference
class Solution:
    def findMinDifference(self, timePoints):
        minutes = [int(x[:2]) * 60 + int(x[3:]) for x in timePoints]
        minutes.sort()
        #modulo returns positive if negative % positive
        #ie. -1249 % 1440 = 191
        #when x in -x % y = ..., modulo does opposite. instead of remainder, it returns the number needed to reach next full value
        #when x in x % -y = ..., instead of remainder, it returns the negative value of -x % y
        #list1 + list2 adds the two together
        #list1 += list2 extends original list
        #list1 = list1 + list2 reassigns list1 to new list
        return min((y - x) % (24 * 60) for x, y in zip(minutes, minutes[1:] + minutes[:1]))

# Q543 Diameter of Binary Tree
class Solution:
    diameter = 0
    def diameterOfBinaryTree(self, root):

        def depth(root):
            left = right = 0

            if root.left:
                left = depth(root.left) + 1
            if root.right:
                right = depth(root.right) + 1

            path = left + right
            self.diameter = max(self.diameter,path)
            return max(left,right)

        if not root:
            return 0

        depth(root)
        return self.diameter

# Q545 Boundary of Binary Tree
class Solution:
    #Two ways of doing bfs:
        #first is with a regular queue
        #second is using a stack to make levels
    def boundaryOfBinaryTree(self, root):
        def dfs_leftmost(node):
            if not node or not node.left and not node.right:
                return
            boundary.append(node.val)
            if node.left:
                dfs_leftmost(node.left)
            else:
                dfs_leftmost(node.right)

        #inorder traversal
        #must be inorder traversal to get elements from left to right
        def dfs_leaves(node):
            if not node:
                return
            dfs_leaves(node.left)
            if node != root and not node.left and not node.right:
                boundary.append(node.val)
            dfs_leaves(node.right)

        def dfs_rightmost(node):
            if not node or not node.left and not node.right:
                return
            if node.right:
                dfs_rightmost(node.right)
            else:
                dfs_rightmost(node.left)
            #add element after recursion is done to put in reverse order
            boundary.append(node.val)

        if not root:
            return []
        boundary = [root.val]
        dfs_leftmost(root.left)
        dfs_leaves(root)
        dfs_rightmost(root.right)
        return boundary

# Q549 Binary Tree Longest Consecutive Sequence II
class Solution:
    def longestConsecutive(self, root):
        def longestConsecutiveHelper(root):
            if not root:
                return 0, 0
            left_len = longestConsecutiveHelper(root.left)
            right_len = longestConsecutiveHelper(root.right)
            cur_inc_len, cur_dec_len = 1, 1
            if root.left:
                if root.left.val == root.val + 1:
                    cur_inc_len = max(cur_inc_len, left_len[0] + 1)
                elif root.left.val == root.val - 1:
                    cur_dec_len = max(cur_dec_len, left_len[1] + 1)
            if root.right:
                if root.right.val == root.val + 1:
                    cur_inc_len = max(cur_inc_len, right_len[0] + 1)
                elif root.right.val == root.val - 1:
                    cur_dec_len = max(cur_dec_len, right_len[1] + 1)
            self.max_len = max(self.max_len, cur_dec_len + cur_inc_len - 1)
            return cur_inc_len, cur_dec_len

        self.max_len = 0
        longestConsecutiveHelper(root)
        return self.max_len
