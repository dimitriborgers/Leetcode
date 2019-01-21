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
