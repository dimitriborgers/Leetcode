# 201 Bitwise AND of Numbers Range


# 202 Happy Number


# 203 Remove Linked List Elements


# 204 Count Primes


# 205 Isomorphic Strings
import collections

class Solution:
    def isIsomorphic(self, s, t):
        seq = collections.defaultdict(set)
        for i in range(len(s)):
            seq[s[i]].add(t[i])
            if len(seq[s[i]]) > 1:
                return False
        return True

# 206 Reverse Linked List


# 207 Course Schedule


# 208 Implement Trie (Prefix Tree)


# 209 Minimum Size Subarray Sum


# 210 Course Schedule II


# 211 Add and Search Word - Data structure design


# 21# 2 Word Search II


# 213 House Robber II


# 214 Shortest Palindrome


# 215 Kth Largest Element in an Array


# 216 Combination Sum III


# 217 Contains Duplicate


# 218 The Skyline Problem
start, end, height = 0, 1, 2
class Solution:
    def getSkyline(self, buildings):
        intervals = self.ComputeSkylineInInterval(buildings, 0, len(buildings))

        res = []
        last_end = -1
        for interval in intervals:
            if last_end != -1 and last_end < interval[start]:
                res.append([last_end, 0])
            res.append([interval[start], interval[height]])
            last_end = interval[end]
        if last_end != -1:
            res.append([last_end, 0])

        return res

    # Divide and Conquer.
    def ComputeSkylineInInterval(self, buildings, left_endpoint, right_endpoint):
        if right_endpoint - left_endpoint <= 1:
            return buildings[left_endpoint:right_endpoint]
        mid = left_endpoint + ((right_endpoint - left_endpoint) / 2)
        left_skyline = self.ComputeSkylineInInterval(buildings, left_endpoint, mid)
        right_skyline = self.ComputeSkylineInInterval(buildings, mid, right_endpoint)
        return self.MergeSkylines(left_skyline, right_skyline)

    # Merge Sort.
    def MergeSkylines(self, left_skyline, right_skyline):
        i, j = 0, 0
        merged = []

        while i < len(left_skyline) and j < len(right_skyline):
            if left_skyline[i][end] < right_skyline[j][start]:
                merged.append(left_skyline[i])
                i += 1
            elif right_skyline[j][end] < left_skyline[i][start]:
                merged.append(right_skyline[j])
                j += 1
            elif left_skyline[i][start] <= right_skyline[j][start]:
                i, j = self.MergeIntersectSkylines(merged, left_skyline[i], i,\
                                                   right_skyline[j], j)
            else: # left_skyline[i][start] > right_skyline[j][start].
                j, i = self.MergeIntersectSkylines(merged, right_skyline[j], j, \
                                                   left_skyline[i], i)

        # Insert the remaining skylines.
        merged += left_skyline[i:]
        merged += right_skyline[j:]
        return merged

    # a[start] <= b[start]
    def MergeIntersectSkylines(self, merged, a, a_idx, b, b_idx):
        if a[end] <= b[end]:
            if a[height] > b[height]:   # |aaa|
                if b[end] != a[end]:    # |abb|b
                    b[start] = a[end]
                    merged.append(a)
                    a_idx += 1
                else:             # aaa
                    b_idx += 1    # abb
            elif a[height] == b[height]:  # abb
                b[start] = a[start]       # abb
                a_idx += 1
            else:  # a[height] < b[height].
                if a[start] != b[start]:                            #    bb
                    merged.append([a[start], b[start], a[height]])  # |a|bb
                a_idx += 1
        else:  # a[end] > b[end].
            if a[height] >= b[height]:  # aaaa
                b_idx += 1              # abba
            else:
                #    |bb|
                # |a||bb|a
                if a[start] != b[start]:
                    merged.append([a[start], b[start], a[height]])
                a[start] = b[end]
                merged.append(b)
                b_idx += 1
        return a_idx, b_idx

# 219 Contains Duplicate II
class Solution:
    def containsNearbyDuplicate(self, nums, k):
        lookup = {}
        for i, num in enumerate(nums):
            if num not in lookup:
                lookup[num] = i
            else:
                if i - lookup[num] <= k:
                    return True
                lookup[num] = i
        return False

# 220 Contains Duplicate III
class Solution:
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        if k < 0 or t < 0:
            return False

        window = dict
        for n in nums:
            if len(window) > k:
                window.popitem(False)

            bucket = n if not t else n // t
            for m in (window.get(bucket - 1), window.get(bucket), window.get(bucket + 1)):
                if m is not None and abs(n - m) <= t:
                    return True
            window[bucket] = n
        return False

# 221 aximal Square


# 222 unt Complete Tree Nodes


# 223 rectangle Area


# 224 basic Calculator


# 225 implement Stack using Queues


# 226 convert Binary Tree


# 227 basic Calculator II


# 228 ummary Ranges


# 229 ajority Element II


# 230 Kth Smallest Element in a BST


# 231 Power of Two


# 23# 2 I plement Queue using Stacks


# 233 Number of Digit One


# 234 Palindrome Linked List


# 235 Lowest Common Ancestor of a Binary Search Tree


# 236 Lowest Common Ancestor of a Binary Tree


# 237 Delete Node in a Linked List


# 238 Product of Array Except Self


# 239 Sliding Window Maximum


# 240 Search a 2D Matrix II#


#241 Different Ways to Add Parentheses


# 242 Valid Anagram


# 243 Shortest Word Distance


# 244 Shortest Word Distance II


# 245 Shortest Word Distance III


# 246 Strobogrammatic Number


# 247 Strobogrammatic Number II


# 248 Strobogrammatic Number III


# 249 Group Shifted Strings


# 250 Count Univalue Subtrees


