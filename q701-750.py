# Q702 Search in a Sorted Array of Unknown Size
class Solution:
    def search(self, reader, target):
        #find the upperbound
        start = 0
        end = 1
        while (reader.get(end) < target) and (reader.get(end) != 2147483647):
            end *= 2
        #Binary search
        while start + 1 < end:
            mid = start + (end - start) // 2
            if reader.get(mid) == target:
                return mid
            elif reader.get(mid) > target:
                end = mid
            else:
                start = mid
        if reader.get(start) == target:
            return start
        if reader.get(end) == target:
            return end
        return -1

# Q703 Kth Largest Element in a Stream
import bisect

class KthLargest:

    def __init__(self, k, nums):
        self.k = k
        self.nums = sorted(nums)
        if len(self.nums) >= self.k:
            self.largest = self.nums[-self.k]
        else:
            self.largest = None

    def add(self, val):
        bisect.insort(self.nums,val)
        if len(self.nums) >= self.k:
            self.largest = self.nums[-self.k]
        else:
            self.largest = None
        return self.largest

# Q715 Range Module
from bisect import bisect_left as bl, bisect_right as br
class RangeModule1:

    def __init__(self):
        self.ivs = []

    def addRange(self, left, right):
        ivs = self.ivs
        ilo, ihi = bl(ivs, left), br(ivs, right)
        if ilo%2 == 1:
            ilo -= 1
            left = ivs[ilo]
        if ihi%2 == 1:
            right = ivs[ihi]
            ihi += 1
        self.ivs = ivs[:ilo] + [left, right] + ivs[ihi:]

    def queryRange(self, left, right):
        ivs = self.ivs
        ilo = br(ivs, left)
        return ilo%2 == 1 and ilo < len(ivs) and ivs[ilo-1] <= left < right <= ivs[ilo]

    def removeRange(self, left, right):
        ivs = self.ivs
        ilo, ihi = bl(ivs, left), br(ivs, right)
        new = []
        if ilo%2 == 1:
            ilo -= 1
            new += [ivs[ilo], left]
        if ihi%2 == 1:
            new += [right, ivs[ihi]]
            ihi += 1
        self.ivs = ivs[:ilo] + new + ivs[ihi:]

from bisect import bisect_left as bl, bisect_right as br

class RangeModule2:

    def __init__(self):
        self._X = []

    def addRange(self, left, right):
        i, j = bl(self._X, left), br(self._X, right)
        #a[1:5] = b -> will shorten or lengthen [1:5] depending on the length of b
        self._X[i:j] = [left]*(i%2 == 0) + [right]*(j%2 == 0)

    def queryRange(self, left, right):
        i, j = br(self._X, left), bl(self._X, right)
        return i == j and i%2 == 1

    def removeRange(self, left, right):
        i, j = bl(self._X, left), br(self._X, right)
        self._X[i:j] = [left]*(i%2 == 1) + [right]*(j%2 == 1)

# Q727 Minimum Window Subsequence
class Solution:
    def minWindow(self, S, T):
        cur = [i if x == T[0] else None
               for i, x in enumerate(S)]

        for j in range(1, len(T)):
            last = None
            new = [None] * len(S)
            for i, u in enumerate(S):
                if last is not None and u == T[j]: new[i] = last
                if cur[i] is not None: last = cur[i]
            cur = new

        ans = 0, len(S)
        for e, s in enumerate(cur):
            if s is not None and e - s < ans[1] - ans[0]:
                ans = s, e
        return S[ans[0]: ans[1]+1] if ans[1] < len(S) else ""

# Q729 My Calendar I
# Brute Force
class MyCalendar:
    def __init__(self):
        self.calendar = []

    def book(self, start, end):
        for s, e in self.calendar:
            if s < end and start < e:
                return False
        self.calendar.append((start, end))
        return True

# Balanced Tree
class Node:
    __slots__ = 'start', 'end', 'left', 'right'
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.left = self.right = None

    def insert(self, node):
        if node.start >= self.end:
            if not self.right:
                self.right = node
                return True
            return self.right.insert(node)
        elif node.end <= self.start:
            if not self.left:
                self.left = node
                return True
            return self.left.insert(node)
        else:
            return False

class MyCalendar:
    def __init__(self):
        self.root = None

    def book(self, start, end):
        if self.root is None:
            self.root = Node(start, end)
            return True
        return self.root.insert(Node(start, end))

# Q731 My Calendar II
class MyCalendarTwo:
    def __init__(self):
        self.calendar = []
        self.overlaps = []

    def book(self, start, end):
        for i, j in self.overlaps:
            if start < j and end > i:
                return False
        for i, j in self.calendar:
            if start < j and end > i:
                self.overlaps.append((max(start, i), min(end, j)))
        self.calendar.append((start, end))
        return True

# Q734 Sentence Similarity
import collections

class Solution:
    def areSentencesSimilar(self, words1, words2, pairs):
        if len(words1) != len(words2):
            return False

        lookup = collections.defaultdict(list)
        for left,right in pairs:
            lookup[left].append(right)
            lookup[left].append(left)
            lookup[right].append(left)
            lookup[right].append(right)

        #more pythonic way of checking something for every element in list
        return all(word1 == word2 or word1 in lookup[word2] or word2 in lookup[word1] for word1,word2 in zip(words1,words2))

class Solution2:
    def areSentencesSimilar(self, words1, words2, pairs):
        if len(words1) != len(words2):
            return False

        lookup = set(tuple(i) for i in pairs)
        return all(w1 == w2 or (w1, w2) in lookup or (w2, w1) in lookup for w1, w2 in zip(words1, words2))

# Q736 Parse Lisp Expression
from operator import add, mul
from collections import deque

class Solution:
    def evaluate(self, expression):

        def scan(s):
            return s.replace('(', ' ( ').replace(')', ' ) ').split()

        def parse(tokens):
            tok = tokens.popleft()
            if tok == '(':
                L = []
                while tokens[0] != ')':
                    L.append(parse(tokens))
                tokens.popleft()
                return L
            else:
                try:
                    return int(tok)
                except ValueError:
                    return tok

        def ast(s):
            return parse(deque(scan(s)))

        def eval(e):
            if not isinstance(e, list):
                if isinstance(e, int):
                    return e
                else:
                    for scope in reversed(env):
                        if e in scope:
                            return scope[e]
            else:
                env.append({})
                if e[0] in ['add', 'mult']:
                    op = add if e[0] == 'add' else mul
                    res = op(eval(e[1]), eval(e[2]))
                else:
                    for i in range(2, len(e), 2):
                        env[-1][e[i-1]] = eval(e[i])
                    res = eval(e[-1])
                env.pop()
                return res

        env = []
        syntax_tree = ast(expression)
        return eval(syntax_tree)

# Q737 Sentence Similarity II
import collections

class Solution:
    def areSentencesSimilarTwo(self, words1, words2, pairs):
        if len(words1) != len(words2):
            return False

        lookup = collections.defaultdict(set)
        for left,right in pairs:
            lookup[left].add(right)
            lookup[right].add(left)

        def dfs(cur,word2):
            if word2 not in cur:
                for word in cur:
                    if word not in visited:
                        visited.add(word)
                        cur = lookup[word]
                        tmp = dfs(cur,word2)
                        if tmp:
                            return True
            else:
                return True

        for word1,word2 in zip(words1,words2):
            if word1 != word2 and word1 not in lookup[word2]:
                visited = {word1,word2}
                if not dfs(lookup[word1],word2):
                    return False
        return True

# Q750 Number Of Corner Rectangles
class Solution:
    def countCornerRectangles(self, grid):
        count = collections.Counter()
        ans = 0
        for row in grid:
            for c1, v1 in enumerate(row):
                if v1:
                    for c2 in range(c1+1, len(row)):
                        if row[c2]:
                            ans += count[c1, c2]
                            count[c1, c2] += 1
        return ans
