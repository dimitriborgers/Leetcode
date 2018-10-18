# Q562 Longest Line of Consecutive One in Matrix
class Solution:
    def longestLine(self, M):
        maxLength = 0
        for i in range(len(M)):
            for j in range(len(M[0])):
                if M[i][j]:
                    maxLength = max(maxLength,max(self.horizontal(i,j,M),self.vertical(i,j,M),self.diagonal_up(i,j,M),self.diagonal_down(i,j,M)))
        return maxLength

    def horizontal(self,i,j,M,count=0):
        while j < len(M[0]) and M[i][j]:
            j += 1
            count += 1
        return count

    def vertical(self,i,j,M,count=0):
        while i < len(M) and M[i][j]:
            i += 1
            count += 1
        return count

    def diagonal_up(self,i,j,M,count=0):
        while j < len(M[0]) and i > 0 and M[i][j]:
            j += 1
            i -= 1
            count += 1
        return count

    def diagonal_down(self,i,j,M,count=0):
        while j < len(M[0]) and i < len(M) and M[i][j]:
            j += 1
            i += 1
            count += 1
        return count
