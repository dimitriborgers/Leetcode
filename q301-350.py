# Q303 Range Sum Query - Immutable
class NumArray:
    def __init__(self, nums):
        self.accu = [0]
        #You can do more that just assignment statements in __init__
        for num in nums:
            self.accu.append(self.accu[-1] + num)

    def sumRange(self, i, j):
        return self.accu[j + 1] - self.accu[i]

# Q308 Range Sum Query 2D - Mutable
from copy import deepcopy

class NumMatrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.dp = deepcopy(matrix)
        for i in range(len(self.dp[0])):
            for j in range(1,len(self.dp)):
                self.dp[j][i] += self.dp[j-1][i]

        for i in range(len(self.dp)):
            for j in range(1,len(self.dp[0])):
                self.dp[i][j] += self.dp[i][j-1]

    def update(self, row, col, val):
        for i in range(row,len(self.dp)):
            self.dp[i][col] -= self.matrix[row][col] - val

        for i in range(row,len(self.dp)):
            for j in range(col+1,len(self.dp[0])):
                self.dp[i][j] -= self.matrix[row][col] - val

    def sumRegion(self, row1, col1, row2, col2):
        total = self.dp[row2][col2] - self.dp[row2][col1-1] - self.dp[row1-1][col2] + self.dp[row1-1][col1-1]
        return total

# Q325 Maximum Size Subarray Sum Equals k
class Solution:
    def maxSubArrayLen(self, nums, k):
        sums = {}
        cur_sum, max_len = 0, 0
        for i in range(len(nums)):
            cur_sum += nums[i]
            if cur_sum == k:
                max_len = i + 1
            elif cur_sum - k in sums:
                max_len = max(max_len, i - sums[cur_sum - k])
            if cur_sum not in sums:
                sums[cur_sum] = i
        return max_len

