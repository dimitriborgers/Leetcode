# Q957 Prison Cells After N Days
class Solution:
    def prisonAfterNDays(self, cells, N):
        N = N % 14
        if not N:
            N = 14
        for _ in range(N):
            cells = [0] + [int(i == j) for i, j in zip(cells, cells[2:])] + [0]
        return cells
