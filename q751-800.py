# Q753 Cracking the Safe
class Solution:
    #Hierholzer's Algorithm
    def crackSafe(self, n, k):
        seen = set()
        ans = []
        #This is a simple way to get all permutations
        #To change to combinations, just make sure set isn't already in seen, this works since {1,2,3} == {1,3,2} (order does not matter)
        def dfs(node):
            #notice how you can use k here since sub-routines can access caller routine variables
            for x in map(str, range(k)):
                nei = node + x
                if nei not in seen:
                    seen.add(nei)
                    dfs(nei[1:])
                    ans.append(x)

        dfs("0" * (n-1))
        return "".join(ans) + "0" * (n-1)

# Q766 Toeplitz Matrix
class Solution:
    def isToeplitzMatrix(self, matrix):
        return all(i == 0 or j == 0 or matrix[i-1][j-1] == val for i, row in enumerate(matrix) for j, val in enumerate(row))

# Q774 Minimize Max Distance to Gas Station
class Solution:
    def minmaxGasDist(self, stations, K):
        def possible(D):
            return sum(int((stations[i+1] - stations[i]) / D)
                       for i in range(len(stations) - 1)) <= K

        lo, hi = 0, 10**8
        while hi - lo > 1e-6:
            mi = (lo + hi) / 2.0
            if possible(mi):
                hi = mi
            else:
                lo = mi
        return lo

# Q776 Split BST

