# Q957 Prison Cells After N Days
class Solution:
    def prisonAfterNDays(self, cells, N):
        N = N % 14
        if not N:
            N = 14
        for _ in range(N):
            cells = [0] + [int(i == j) for i, j in zip(cells, cells[2:])] + [0]
        return cells

# Q973 K Closest Points to Origin
class Solution:
    def kClosest(self, points, K):

        def distance(x,y):
            return math.sqrt(x**2+y**2)

        tmp = sorted([(distance(x,y),[x,y]) for x,y in points])
        return [loc for pos,loc in tmp][:K]

# Q975 Odd Even Jump
class Solution:
    def oddEvenJumps(self, A):
        N = len(A)

        def make(B):
            ans = [None] * N
            stack = []
            for i in B:
                while stack and i > stack[-1]:
                    ans[stack.pop()] = i
                stack.append(i)
            return ans

        B = sorted(range(N), key = lambda i: A[i])
        oddnext = make(B)
        B.sort(key = lambda i: -A[i])
        evennext = make(B)

        odd = [False] * N
        even = [False] * N
        odd[N-1] = even[N-1] = True

        for i in range(N-2, -1, -1):
            if oddnext[i] is not None:
                odd[i] = even[oddnext[i]]
            if evennext[i] is not None:
                even[i] = odd[evennext[i]]

        return sum(odd)
