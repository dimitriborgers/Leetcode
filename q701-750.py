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
        words1 = words1.split(' ')
        words2 = words2.split(' ')

        if len(words1) != len(words2):
            return False

        lookup = collections.defaultdict(list)
        for left,right in pairs:
            lookup[left].append(right)
            lookup[left].append(left)
            lookup[right].append(left)
            lookup[right].append(right)

        #more pythonic way of checking something for every element in list
        return all(word1 in lookup[word2] and word2 in lookup[word1] for word1,word2 in zip(words1,words2))

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
