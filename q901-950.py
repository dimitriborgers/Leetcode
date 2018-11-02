# Q900 RLEIterator
class RLEIterator:

    def __init__(self, A):
        self.seq = []
        self.index = -1
        for i in range(len(A)-1):
            if not i % 2:
                self.seq.extend([A[i+1]]*A[i])
        self.remainder = len(self.seq)

    def next(self, n):
        if n > self.remainder:
            self.index += n
            self.remainder -= n
            return -1

        self.index += n
        self.remainder -= n
        return self.seq[self.index]

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
class Solution:
    def totalFruit(self, tree):
        maxlength = float('-inf')
        for i in range(len(tree)-1):
            j = i+1
            basket = {tree[i]}
            while j < len(tree):
                if tree[j] not in basket:
                    basket.add(tree[j])
                if len(basket) > 2:
                    break
                j += 1
            maxlength = max(j-i,maxlength)
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

# Q911 Online Election - Didn't understand
class TopVotedCandidate:

    def __init__(self, persons, times):
        self.A = []
        self.count = collections.Counter()
        for p, t in zip(persons, times):
            self.count[p] = c = self.count[p] + 1
            while len(self.A) <= c: self.A.append([])
            self.A[c].append((t, p))

    def q(self, t):
        lo, hi = 1, len(self.A)
        while lo < hi:
            mi = (lo + hi) / 2
            if self.A[mi][0][0] <= t:
                lo = mi + 1
            else:
                hi = mi
        i = lo - 1
        j = bisect.bisect(self.A[i], (t, float('inf')))
        return self.A[i][j-1][1]
