# Q524 Longest Word in Dictionary through Deleting
class Solution:
    def findLongestWord(self, s, d):
        def searcher(s,word):
            if not word:
                return True
            if word[0] in s:
                return searcher(s[s.find(word[0])+1:],word[1:])
            return False

        result = [(len(word),word) for word in d if searcher(s,word)]
        length = max(result,key=lambda x:x[0])[0]
        return min([word for num,word in result if num == length])


# Q528 Random Pick with Weight
#Space: O(1)   Time: O(n)
import random
import bisect

class Solution:

    def __init__(self, w):
        self.__prefix_sum = list(w)
        for i in range(1, len(w)):
            self.__prefix_sum[i] += self.__prefix_sum[i-1]

    def pickIndex(self):
        target = random.randint(0, self.__prefix_sum[-1]-1)
        #gives the index
        #bisect == bisect_right: if you have [1,1,1], index will be 3
        #bisect_left: if you have [1,1,1], index will be 0. So you actually take the index of the first instance found
        #if element you're inputting isn't in list already, then both methods should return the same index
        return bisect.bisect_right(self.__prefix_sum, target)

# Q539 Minimum Time Difference
class Solution:
    def findMinDifference(self, timePoints):
        minutes = [int(x[:2]) * 60 + int(x[3:]) for x in timePoints]
        minutes.sort()
        #modulo returns positive if negative % positive
        #ie. -1249 % 1440 = 191
        #when x in -x % y = ..., modulo does opposite. instead of remainder, it returns the number needed to reach next full value
        #when x in x % -y = ..., instead of remainder, it returns the negative value of -x % y
        #list1 + list2 adds the two together
        #list1 += list2 extends original list
        #list1 = list1 + list2 reassigns list1 to new list
        return min((y - x) % (24 * 60) for x, y in zip(minutes, minutes[1:] + minutes[:1]))

# Q549 Binary Tree Longest Consecutive Sequence II
class Solution:
    def longestConsecutive(self, root):
        def longestConsecutiveHelper(root):
            if not root:
                return 0, 0
            left_len = longestConsecutiveHelper(root.left)
            right_len = longestConsecutiveHelper(root.right)
            cur_inc_len, cur_dec_len = 1, 1
            if root.left:
                if root.left.val == root.val + 1:
                    cur_inc_len = max(cur_inc_len, left_len[0] + 1)
                elif root.left.val == root.val - 1:
                    cur_dec_len = max(cur_dec_len, left_len[1] + 1)
            if root.right:
                if root.right.val == root.val + 1:
                    cur_inc_len = max(cur_inc_len, right_len[0] + 1)
                elif root.right.val == root.val - 1:
                    cur_dec_len = max(cur_dec_len, right_len[1] + 1)
            self.max_len = max(self.max_len, cur_dec_len + cur_inc_len - 1)
            return cur_inc_len, cur_dec_len

        self.max_len = 0
        longestConsecutiveHelper(root)
        return self.max_len
