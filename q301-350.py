# Q303 Range Sum Query - Immutable
class NumArray:
    def __init__(self, nums):
        self.accu = [0]
        #You can do more that just assignment statements in __init__
        for num in nums:
            self.accu.append(self.accu[-1] + num)

    def sumRange(self, i, j):
        return self.accu[j + 1] - self.accu[i]

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

