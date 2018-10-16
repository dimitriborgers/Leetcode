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
