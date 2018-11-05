# Q679 24 Game
#self-made classes naturally have __hash__ function made for them
#heapq methods don't work with self-made classes, unless you define le,lt,etc.
from operator import truediv, mul, add, sub

class Solution:
    def judgePoint24(self, A):
        if not A:
            return False
        if len(A) == 1:
            return A[0] - 24 == 0

        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    B = [A[k] for k in range(len(A)) if i != k != j]
                    for op in (truediv, mul, add, sub):
                        #Makes sure you don't divide by 0
                        if op is not truediv or A[j]:
                            B.append(op(A[i], A[j]))
                            if self.judgePoint24(B):
                                return True
                            B.pop()
        return False

# Q681 Next Closest Time
class Solution:
    def nextClosestTime(self, time):
        #cannot do [::-1] or anything of that sort in set(). It is not suscriptable
        time = [int(i) for i in time if i != ':']

        for j in range(time[-1]+1,10):
            if j in time and j != time[-1]:
                time[-1] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

        for j in range(time[-2]+1,6):
            if j in time and j != time[-2]:
                time[-2] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

        for j in range(time[-3]+1,5):
            if j in time and j != time[-3]:
                time[-3] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

        for j in range(time[-4]+1,3):
            if j in time and j != time[-4]:
                time[-4] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

        minimum = min(time)
        time = [minimum]*4
        time = ''.join(str(i) for i in time)
        #concatenation
        return time[:2]+':'+time[2:]

    def nextClosestTime2(self, time):
        cur = 60 * int(time[:2]) + int(time[3:])
        allowed = {int(x) for x in time if x != ':'}
        while True:
            cur = (cur + 1) % (24 * 60)
            #automatic packing if multiple values assigned to one variable
            #second forloop done first
            if all(digit in allowed for block in divmod(cur, 60) for digit in divmod(block, 10)):
                #2 puts two spaces of padding minimum and 0 puts a 0 in front of the number if only 1 is present
                #d is just for signed integer decimal
                return "{:02d}:{:02d}".format(*divmod(cur, 60))

# Q682


# Q683 K Empty Slots
import bisect

class Solution:
    def kEmptySlots(self, flowers, k):
        days = [0] * len(flowers)
        for day, position in enumerate(flowers, 1):
            days[position - 1] = day

        ans = float('inf')
        left, right = 0, k+1
        while right < len(days):
            for i in range(left + 1, right):
                if days[i] < days[left] or days[i] < days[right]:
                    left, right = i, i+k+1
                    break
            #else statement will occur after end of loop. Will not occur if break statement is encountered.
            else:
                ans = min(ans, max(days[left], days[right]))
                left, right = right, right+k+1

        return ans if ans < float('inf') else -1

    def kEmptySlots2(self, flowers, k):
        active = []
        #second arg in enumerate lets you choose what number to start enumeration on.
        for day, flower in enumerate(flowers, 1):
            #bisect.bisect tells you where exactly the value would be inserted
            i = bisect.bisect(active, flower)
            #i-(i>0) -> i>0 is 1 if true, 0 if not
            '''
            seq = []
            for i in seq[2:4]
            This does not give a out of bounds error. It will just skip this loop if the range given does not apply
            To start loop, the first number must be accurate. Loop exits if reached end or number no longer accurate
            '''
            for neighbor in active[i-(i>0):i+1]:
                if abs(neighbor - flower) - 1 == k:
                    return day
            #bisect.insort knows where to insert element
            #moves element previous at position to the right
            bisect.insort(active,flower)
        return -1

# Q684 Redundant Connection
import collections

class Solution1:
    def findRedundantConnection(self, edges):
        graph = collections.defaultdict(set)

        def dfs(source, target):
            if source not in seen:
                seen.add(source)
                if source == target:
                    return True
                return any(dfs(nei, target) for nei in graph[source])

        for u, v in edges:
            #notice that seen does not have to be self.seen since dfs is accessing an already created public element
            seen = set()
            if u in graph and v in graph and dfs(u, v):
                return u, v
            graph[u].add(v)
            graph[v].add(u)

class Solution2:
    def findRedundantConnection(self, edges):
        dsu = DSU()
        for edge in edges:
            if not dsu.union(*edge):
                return edge

class DSU:
    def __init__(self):
        #creates range class, can be accessed by index.
        #range class is not same as generator, generators are not suscriptable, which means they can't be accessed by index.
        self.parent = range(1001)
        self.rank = [0] * 1001

    def find(self, x):
        if self.parent[x] != x:
            #path compression
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    #union by rank
    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        elif self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True

# Q686 Repeated String Match
class Solution:
    def repeatedStringMatch(self, A, B):
        if len(B) < len(A):
            return -1

        q = ((len(B)-1) // len(A)) + 1
        for i in range(2):
            if B in A * (q+i):
                return q+i
        return -1
