# Q901 Online Stock Span - Didn't understand
class StockSpanner:

    def __init__(self):
        self.__s = []

    def next(self, price):
        result = 1
        while self.__s and self.__s[-1][0] <= price:
            result += self.__s.pop()[1]
        self.__s.append([price, result])
        return result

# Q902 Numbers At Most N Given Digit Set
import itertools

class Solution:
    result = []
    def atMostNGivenDigitSet(self, D, N):
        for i in range(1,len(D)):
            self.result.extend(list(itertools.permutations(D,i)))
        self.result = [''.join(i) for i in self.result if int(''.join(i)) <= N]
        addition = []
        for i in D:
            for j in range(1,len(str(N))+1):
                addition.append(i*j)
        addition = [i for i in addition if int(i) <= N]
        self.result.extend(addition)
        return sorted(set(self.result))

# 903 Valid Permutations for DI Sequence - Didn't undestand
from functools import lru_cache

class Solution:
    def numPermsDISequence(self, S):
        MOD = 10**9 + 7
        N = len(S)

        @lru_cache(None)
        def dp(i, j):
            # How many ways to place P_i with relative rank j?
            if i == 0:
                return 1
            elif S[i-1] == 'D':
                return sum(dp(i-1, k) for k in range(j, i)) % MOD
            else:
                return sum(dp(i-1, k) for k in range(j)) % MOD

        return sum(dp(N, j) for j in range(N+1)) % MOD

# Q904 Fruit Into Baskets
from collections import defaultdict

class Solution:
    def totalFruit(self, tree):
        maxlength = 1
        i = j = 0
        basket = set()
        lookup = defaultdict(int)
        while i < len(tree):
            basket.add(tree[i])
            lookup[tree[i]] += 1
            if len(basket) > 2:
                while len(basket) > 2:
                    lookup[tree[j]] -= 1
                    if lookup[tree[j]] == 0:
                        basket.remove(tree[j])
                        del lookup[tree[j]]
                    j += 1
            i += 1
            maxlength = max(maxlength,i-j)

        return maxlength

# Q905 Sort Array By Parity
class Solution:
    def sortArrayByParity2(self, A):
        #in place sort using 2 pointers
        i, j = 0, len(A) - 1
        while i < j:
            if A[i] % 2 > A[j] % 2:
                A[i], A[j] = A[j], A[i]

            if A[i] % 2 == 0: i += 1
            if A[j] % 2 == 1: j -= 1

        return A

    def sortArrayByParity2(self, A):
        #can add to list together with a plus sign
        #difference between + and .extend is that plus can return a value, while extend does not
        return ([x for x in A if x % 2 == 0] +
                [x for x in A if x % 2 == 1])

# Q906 Super Palindromes
class Solution:
    def superpalindromesInRange(self, L, R):
        result = []
        for i in range(1000):
            if str(i) == str(i)[::-1] and int(L)<=i**2<int(R) and str(i**2) == str(i**2)[::-1]:
                result.append(i**2)
            if i**2 > int(R):
                break
        return result

# Q907 Sum of Subarray Minimums
class Solution:
    def sumSubarrayMins(self, A):
        #to make the subsets hashable, you have to make them tuples. However, to do this you have to do tuple(). just using () will make it a generator, not a tuple
        return [tuple(x for (pos,x) in enumerate(A) if (2**pos) & a)for a in range(2**len(A))]

# Q908 Smallest Range I
class Solution:
    def smallestRangeI(self, A, K):
        return max(0, max(A) - min(A) - 2*K)

# Q909 Snakes and Ladders


# Q910 Smallest Range II
class Solution:
    def smallestRangeII(self, A, K):
        A.sort()
        mi, ma = A[0], A[-1]
        ans = ma - mi
        for i in range(len(A) - 1):
            a, b = A[i], A[i+1]
            ans = min(ans, max(ma-K, a+K) - min(mi+K, b-K))
        return ans

# Q911 Online Election
import bisect

class TopVotedCandidate:

    def __init__(self, persons, times):
        self.persons = persons
        self.times = times
        self.winners = []
        winner = persons[0],0
        for pos,e in enumerate(persons):
            winner = (winner[0],winner[1]) if winner[0] >= e else (e,pos)
            self.winners.append(winner[1])

    def q(self, t):
        location = bisect.bisect(self.times,t)
        return self.winners[location-1]

# Q929 Unique Email Addresses
import itertools,collections

class Solution:
    def numUniqueEmails(self, emails):
        lookup = collections.defaultdict(int)
        for email in emails:
            email = ''.join(word for word in list(itertools.takewhile(lambda x : x != '+',email)) if word != '.') + '@' + ''.join(itertools.takewhile(lambda x : x != '@',reversed(email)))
            lookup[email] += 1

        return (len(lookup))

# Q947 Most Stones Removed with Same Row or Column
from collections import Counter

class DSU:

    def __init__(self,stones):
        self.parent = [i for i in range(len(stones))]
        self.rank = [1]*len(stones)

    def find(self,value):
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self,left,right):
        parentLeft = self.find(left)
        parentRight = self.find(right)
        if parentLeft != parentRight:
            if self.rank[parentLeft] > self.rank[parentRight]:
                self.parent[parentRight] = parentLeft
            elif self.rank[parentLeft] < self.rank[parentRight]:
                self.parent[parentLeft] = parentRight
            else:
                self.parent[parentRight] = parentLeft
                self.rank[parentLeft] += 1

    def max(self):
        print(Counter(self.parent))
        return sum(i-1 for i in Counter(self.parent).values())

class Solution:
    def removeStones(self, stones):
        dsu = DSU(stones)
        for i in range(len(stones)):
            for j in range(len(stones)):
                if self.check(stones[i],stones[j]):
                    dsu.union(i,j)
                dsu.find(i)
                dsu.find(j)
        return dsu.max()

    def check(self,left,right):
        if left[0] == right[0] or left[1] == right[1]:
            return True
        return False
