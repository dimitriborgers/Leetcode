# Q853 Car Fleet
class Solution:
    def carFleet(self, target, position, speed):
        cars = sorted(zip(position, speed))
        times = [float(target - pos) / spe for pos, spe in cars]
        ans = 0
        while len(times) > 1:
            lead = times.pop()
            if lead < times[-1]:
                ans += 1
            else:
                times[-1] = lead

        #Adding True in math equation equates to 1
        return ans + bool(times)

# Q855 Exam Room
class ExamRoom:
    def __init__(self, N):
        self.N = N
        self.students = []

    def seat(self):
        if not self.students:
            student = 0
        else:
            #Good way to use enumerate and comparing two indexes without going out of bounds. First, store 0th index, and then when you enumerate through, don't compare if pos == 0.
            dist, student = self.students[0], 0
            for pos, s in enumerate(self.students):
                if pos:
                    prev = self.students[pos-1]
                    d = (s - prev) // 2
                    if d > dist:
                        dist, student = d, prev + d

            d = self.N - 1 - self.students[-1]
            if d > dist:
                student = self.N - 1

        bisect.insort(self.students, student)
        return student

    def leave(self, p):
        self.students.remove(p)

# 857 Minimum Cost to Hire K Workers
import heapq

class Solution:
    def mincostToHireWorkers(self, quality, wage, K):
        workers = sorted([[w/q, q] for w, q in zip(wage, quality)])
        result = float("inf")
        qsum = 0
        max_heap = []
        for r, q in workers:
            qsum += q
            #adding negative q to make it max heap
            heapq.heappush(max_heap, -q)
            if len(max_heap) > K:
                qsum -= -heapq.heappop(max_heap)
            if len(max_heap) == K:
                result = min(result, qsum*r)
        return result

# Q863 All Nodes Distance K in Binary Tree
class Solution:
    def distanceK(self, root, target, K):
        def dfs(node, par = None):
            if node:
                #you can add attributes to a class instance
                node.par = par
                dfs(node.left, node)
                dfs(node.right, node)

        dfs(root)

        queue = collections.deque([(target, 0)])
        visited = {target}
        while queue:
            if queue[0][1] == K:
                return [node.val for node, d in queue]
            node, d = queue.popleft()
            for nei in (node.left, node.right, node.par):
                if nei and nei not in visited:
                    visited.add(nei)
                    queue.append((nei, d+1))

        return []

# Q900 RLEIterator
# Space limit exceeded for self.seq array when using giant numbers.
class RLEIterator1:

    def __init__(self, A):
        self.seq = []
        self.index = -1
        for i in range(len(A)-1):
            if not i % 2:
                self.seq.extend([A[i+1]]*A[i])
        self.remainder = len(self.seq)

    def next(self, n):
        if n > self.remainder:
            self.index += n
            self.remainder -= n
            return -1

        self.index += n
        self.remainder -= n
        return self.seq[self.index]

class RLEIterator2:

    def __init__(self, A):
        self.inputList = A

    def next(self, n):
        index = 0
        while index < len(self.inputList):
            if self.inputList[index] == 0:
                self.inputList = self.inputList[index+2:]
            elif n <= self.inputList[index]:
                output = self.inputList[index+1]
                self.inputList[index] -= n
                return output
            else:
                n -= self.inputList[index]
                self.inputList = self.inputList[index+2:]
        if n > 0:
            return -1
