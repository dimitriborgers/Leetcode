# Q252 Meeting Rooms
class Solution:
    def canAttendMeetings(self, intervals):
        #if intervals.sort() doesn't work, just specify the key to use
        intervals.sort(key=lambda x: x.start)

        for i in range(1, len(intervals)):
            if intervals[i].start < intervals[i-1].end:
                return False

        return True

# Q253 Meeting Rooms II
class Solution1:
    def minMeetingRooms(self, intervals):
        if not intervals:
            return 0
        array = [0]*(max(x.end for x in intervals)+1)
        for interval in intervals:
            array[interval.start] += 1
            array[interval.end] -= 1

        sum = 0
        for i in range(len(array)):
            sum += array[i]
            array[i] = sum

        return max(array)

class Solution2:
    def minMeetingRooms(self, intervals):
        intervals.sort(key=lambda x:x.start)
        heap = []
        for i in intervals:
            if heap and i.start >= heap[0]:
                #Means two intervals can use the same room
                #Pop and return the smallest item from the heap, and also push the new item.
                heapq.heapreplace(heap, i.end)
            else:
                #New room is allocated
                heapq.heappush(heap, i.end)
        return len(heap)

# Q265 Paint House II
class Solution:
    def minCostII(self, costs):
        if not costs:
            return 0
        n, k = len(costs), len(costs[0])
        for i in range(1, n):
            min1 = min(costs[i-1])
            idx = costs[i-1].index(min1)
            min2 = min(costs[i-1][:idx] + costs[i-1][idx+1:])
            for j in range(k):
                if j == idx:
                    costs[i][j] += min2
                else:
                    costs[i][j] += min1
        return min(costs[-1])

# Q271 Encode and Decode Strings
from itertools import dropwhile

class Codec:

    def encode(self, strs):
        encoded_str = ''
        for s in strs:
            #How to add 0's to fill gap
            encoded_str += str(len(s)).zfill(8) + s
        return encoded_str

    def decode(self, s):
        i = 0
        strs = []
        while i < len(s):
            l = int(''.join(list(dropwhile(lambda x: x == '0',s[i:i+8]))))
            strs.append(s[i+8:i+8+l])
            i += 8+l
        return strs

# Q273 Integer to English Words
class Solution:
    def numberToWords(self, num):
        if num == 0:
            return "Zero"

        lookup = {0: "Zero", 1:"One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen", 16: "Sixteen", 17: "Seventeen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty", 30: "Thirty", 40: "Forty", 50: "Fifty", 60: "Sixty", 70: "Seventy", 80: "Eighty", 90: "Ninety"}

        unit = ["", "Thousand", "Million", "Billion"]

        res, i = [], 0
        while num:
            cur = num % 1000
            if num % 1000:
                res.append(self.threeDigits(cur, lookup, unit[i]))
            num //= 1000
            i += 1
        return " ".join(res[::-1])

    def threeDigits(self, num, lookup, unit):
        res = []
        if num // 100:
            res = [lookup[num // 100] + " " + "Hundred"]
        if num % 100:
            res.append(self.twoDigits(num % 100, lookup))
        if unit != "":
            res.append(unit)
        return " ".join(res)

    def twoDigits(self, num, lookup):
        if num in lookup:
            return lookup[num]
        return lookup[(num // 10) * 10] + " " + lookup[num % 10]

# Q284 Peeking Iterator
class PeekingIterator:
    def __init__(self, iterator):
        self.iter = iterator
        self.temp = self.iter.next() if self.iter.hasNext() else None

    def peek(self):
        return self.temp

    def next(self):
        ret = self.temp
        self.temp = self.iter.next() if self.iter.hasNext() else None
        return ret

    def hasNext(self):
        return self.temp is not None

# Q285 Inorder Successor in BST
class Solution:
    def inorderSuccessor(self, root, p):
        # If it has right subtree.
        if p and p.right:
            p = p.right
            while p.left:
                p = p.left
            return p

        # Search from root.
        successor = None
        while root and root != p:
            if root.val > p.val:
                successor = root
                root = root.left
            else:
                root = root.right

        return successor

# Q288 Unique Word Abbreviation
# Doesn't work
class ValidWordAbbr:

    def __init__(self, dictionary):
        self.ori = list(dictionary)
        self.dic = list(dictionary)
        for pos,e in enumerate(self.dic):
            if len(e) > 2:
                self.dic[pos] = e.replace(e[1:-1],str(len(e[1:-1])))


    def isUnique(self, word):
        tmp = word.replace(word[1:-1],str(len(word[1:-1])))
        for i in range(len(self.dic)):
            if self.ori[i] == word:
                continue
            if self.dic[i] == tmp:
                return False
        return True

class ValidWordAbbr1:
    def __init__(self, arr):
        self.dic = {}
        for word in set(arr):
            abbr = self.abbrev(word)
            if abbr not in self.dic:
                self.dic[abbr] = word
            else:
                self.dic[abbr] = False

    def isUnique(self, word):
            abbr = self.abbrev(word)
            if abbr not in self.dic:
                return True
            else:
                return self.dic[abbr] == word

    def abbrev(self, word):
        if len(word) < 3:
            return word
        else:
            return word[0] + str(len(word)-2) + word[-1]

# Q289 Game of Life
class Solution:
    def gameOfLife(self, board):

        directions = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]

        def check(r,c):
            return 0<=r<len(board) and 0<=c<len(board[0])

        def outcome(r,c):
            total = sum(board[r+d[0]][c+d[1]] for d in directions if check(r+d[0],c+d[1]))
            if board[r][c] == 1:
                if total <= 1:
                    return 0
                elif 2 <= total <= 3:
                    return 1
                elif total >= 4:
                    return 0
            else:
                if total == 3:
                    return 1
                else:
                    return 0

        #how to create a 2d array using list comprehension
        board[:] = [[outcome(r,c) for c,col in enumerate(row)] for r,row in enumerate(board)]

# Q295 Find Median from Data Stream
import bisect

class MedianFinder:

    def __init__(self):
        self.is_even = True
        self.sequence = []

    def __repr__(self):
        return '{}'.format(self.sequence)

    def addNum(self, num):
        if self.is_even:
            self.is_even = False
        else:
            self.is_even = True
        bisect.insort(self.sequence,num)

    def findMedian(self):
        if not self.sequence:
            return None

        if self.is_even:
            return (self.sequence[len(self.sequence) // 2 - 1] + self.sequence[len(self.sequence) // 2]) / 2
        else:
            return self.sequence[len(self.sequence) // 2]

# Q297 Serialize and Deserialize Binary Tree
from collections import deque

class Codec:

    def serialize(self, root):
        queue = deque([root])
        result = []
        while queue:
            cur = queue.popleft()
            if not cur:
                result.append(None)
            else:
                result.append(cur.val)
                queue.append(cur.left)
                queue.append(cur.right)
        return str(result)

    def deserialize(self, data):
        data = deque(i.strip() for i in data[1:-1].split(","))
        val = data.popleft()
        root = None if val == 'None' else TreeNode(int(val))
        queue = deque([root])
        while queue:
            cur = queue.popleft()
            if cur:
                a, b = data.popleft(), data.popleft()
                cur.left = TreeNode(int(a)) if a != "None" else None
                cur.right = TreeNode(int(b)) if b != "None" else None
                queue.append(cur.left)
                queue.append(cur.right)
        return root

# Q299 Bulls and Cows
from collections import defaultdict

class Solution:
    def getHint(self, secret, guess):
        secretSet = defaultdict(int)
        guessSet = defaultdict(int)
        bulls = cows = i = 0

        while i < len(secret):
            if secret[i] == guess[i]:
                secret = secret[:i]+secret[i+1:]
                guess = guess[:i]+guess[i+1:]
                bulls += 1
            else:
                secretSet[secret[i]] += 1
                guessSet[guess[i]] += 1
                i += 1
        for k,v in secretSet.items():
            if k in guessSet:
                cows += min(secretSet[k],guessSet[k])
        return '{}A{}B'.format(bulls,cows)

