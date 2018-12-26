# Q403 Frog Jump
# Time Limit Exceeded
class Solution1:
    def canCross(self, stones):
        destination = stones[-1]

        def possibilities(cur,k):
            if cur > len(stones):
                return False

            if stones[cur] == destination:
                return True

            for jump in range(k-1,k+2):
                nxt = stones[cur] + jump
                if nxt in stones[cur+1:]:
                    #second argument let's you start search at an index
                    location = stones.index(nxt,cur)
                    result = possibilities(location,jump)
                    if result:
                        return True

            return False

        return possibilities(stones[0],0)

class Solution2:
    def canCross(self, stones):
        if stones[1] != 1:
            return False
        d = dict((x,set()) for x in stones)
        d[1].add(1)
        for i in range(len(stones[1:])):
            for j in d[stones[i]]:
                for k in range(j-1, j+2):
                    if k > 0 and stones[i]+k in d:
                        d[stones[i]+k].add(k)

        return d[stones[-1]] != set()

# Q410 Split Array Largest Sum
class Solution:
    def splitArray(self, nums, m):
        L, R = 0, sum(nums) + 1

        ans = 0
        while L < R:
            mid = (L + R) // 2
            if self.guess(mid, nums, m):
                ans = mid
                R = mid
            else:
                L = mid + 1
        return ans

    @staticmethod
    def guess(mid, nums, m):
        sum = 0
        for i in range(0, len(nums)):
            if sum + nums[i] > mid:
                m -= 1
                sum = nums[i]
                if nums[i] > mid:
                    return False
            else:
                sum += nums[i]
        return m >= 1

# Q418 Sentence Screen Fitting
# Time limit exceeded
class Solution1:
    def wordsTyping(self, sentence, rows, cols):
        self.count = 0
        self.word_counter = 0
        self.words = len(sentence)
        for _ in range(rows):
            self.sequence = []
            self.recur(0,sentence[self.word_counter],cols)
        return self.count

    def recur(self,index,word,length):
        if index + len(word) <= length:
            self.sequence.extend(list(word))
            self.sequence.append('-')
            index += len(word) + 1

            if self.word_counter == self.words - 1:
                self.count += 1

            self.word_counter = (self.word_counter + 1) % self.words
            self.recur(index,sentence[self.word_counter],length)

class Solution2:
    def wordsTyping(self, sentence, rows, cols):
        s = ' '.join(sentence) + ' '
        start = 0
        for i in range(rows):
            start += cols - 1
            if s[start % len(s)] == ' ':
                start += 1
            elif s[(start + 1) % len(s)] == ' ':
                start += 2
            else:
                while start > 0 and s[(start - 1) % len(s)] != ' ':
                    start -= 1
        return start // len(s)

# Q419 Battleships in a Board
class Solution:
    def countBattleships(self, board):
        prev, count = False,0

        for r,row in enumerate(board):
            for c,col in enumerate(row):
                if board[r][c] == 'X' and ((r > 0 and board[r-1][c] == 'X') or (r < len(board)-1 and board[r+1][c] == 'X')):
                    if r == 0 or board[r-1][c] != 'X':
                        count += 1
                else:
                    if prev == True and board[r][c] != 'X':
                        prev = False
                        count += 1
                    elif c == len(board[0])-1 and board[r][c] == 'X':
                        prev = False
                        count += 1
                    elif board[r][c] == 'X':
                        prev = True
                    else:
                        continue

        return count

# Q424 Longest Repeating Character Replacement
#ord('A') - 65
#ord('a') - 97
#Sliding window
class Solution:
    def characterReplacement(self, s, k):
        res = 0

        cnts = [0] * 26
        times, i, j = k, 0, 0
        while j < len(s):
            cnts[ord(s[j]) - ord('A')] += 1
            if s[j] != s[i]:
                times -= 1
                if times < 0:
                    res = max(res, j - i)
                    while i < j and times < 0:
                        cnts[ord(s[i]) - ord('A')] -= 1
                        i += 1
                        times = k - (j - i + 1 - cnts[ord(s[i]) - ord('A')])
            j += 1

        return max(res, j - i + min(i, times))

# Q428 Serialize and Deserialize N-ary Tree
class Node:
    def __init__(self, val, children):
        self.val = val
        self.children = children

class Codec:

    #store an ‘end of children’ marker with every node.
    def serialize(self, root):
        def dfs(node):
            if not node:
                return
            vals.append(str(node.val))
            for child in node.children:
                dfs(child)
            vals.append("#")

        vals = []
        dfs(root)
        return " ".join(vals)


    def deserialize(self, data):
        #split is not actually needed
        #to not use it, remove the ' ' (space) in between the numbers in the serialized data, and then just call on: return dfs(iter(data))
        def isplit(source, sep):
            sepsize = len(sep)
            start = 0
            while True:
                idx = source.find(sep, start)
                #find returns -1 if not found
                if idx == -1:
                    yield source[start:]
                    #return statement acts as a stop iteration
                    return
                yield source[start:idx]
                start = idx + sepsize

        def dfs(vals):
            val = next(vals)
            if val == "#":
                return None
            root = Node(int(val), [])
            child = dfs(vals)
            while child:
                root.children.append(child)
                child = dfs(vals)
            return root

        if not data:
            return None

        #iter creates an iterator from iterable
        #iter can make an iterator from a string
        return dfs(iter(isplit(data, ' ')))

# Q444 Sequence Reconstruction
class Solution1:
    def sequenceReconstruction(self, org, seqs):
        pos = {org[i] : i for i in range(len(org))}
        flag = [0] * len(org)
        stupid_test = 0

        to_match = len(org) - 1
        for seq in seqs:
            stupid_test += len(seq)
            for i in range(len(seq)):
                if seq[i] <= 0 or seq[i] > len(org):
                    return False
                if i == 0:
                    continue
                x = seq[i - 1]
                y = seq[i]
                if pos[x] >= pos[y]:
                    return False
                if pos[x] + 1 == pos[y] and flag[x-1] == 0:
                    flag[x-1] = 1
                    to_match -= 1
        return to_match == 0 and stupid_test != 0

class Solution2:
    def sequenceReconstruction(self, org, seqs):
        order, orders, graph, seen = collections.defaultdict(int), set(), collections.defaultdict(set), set()
        for seq in seqs:
            for i in range(len(seq)):
                if i > 0:
                    if seq[i] == seq[i - 1]:
                        return False
                    graph[seq[i - 1]].add(seq[i])
                seen.add(seq[i])
        if not seen:
            return False
        for i in range(len(org) - 1, -1, -1):
            if org[i] in seen:
                seen.discard(org[i])
            order[org[i]] = max([order[v] for v in graph[org[i]]] or [0]) + 1
            before = set(v for v in graph[org[i]] if v in seen)
            if order[org[i]] in orders or before:
                return False
            orders.add(order[org[i]])
        return not seen

# Q449 Serialize and Deserialize BST
