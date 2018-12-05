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
