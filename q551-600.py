# Q562 Longest Line of Consecutive One in Matrix
class Solution1:
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

class Solution2:
    def longestLine(self, M):
        if len(M)==0: return 0
        m,n=len(M),len(M[0])
        P=[[[0]*4 for _ in range(0,n)] for j in range(0,m)]

        for r in range(m):
            for c in range(n):
                if c==0: P[r][c][0]=M[r][c]
                else: P[r][c][0]= (P[r][c-1][0]+M[r][c])*M[r][c]

                if r==0: P[r][c][1]=M[r][c]
                else: P[r][c][1]= (P[r-1][c][1]+M[r][c])*M[r][c]

                if r==0 or c==n-1: P[r][c][2]=M[r][c]
                else: P[r][c][2]= (P[r-1][c+1][2]+M[r][c])*M[r][c]

                if r==0 or c==0: P[r][c][3]=M[r][c]
                else: P[r][c][3]= (P[r-1][c-1][3]+M[r][c])*M[r][c]

        return max([P[i][j][k] for i in range(0,m) for j in range(0,n) for k in range(0,4)])

# Q568 Maximum Vacation Days
class Solution:
    def maxVacationDays(self, flights, days):

        for i in range(len(flights)):
            for j in range(len(flights[0])):
                if i == j:
                    flights[i][j] = 1

        def dfs(current,day):
            if day == -1:
                return max(dfs(city,day+1) for city,name in enumerate(flights[current]) if name)

            if day == len(days[0])-1:
                return days[current][day]

            return days[current][day] + max(dfs(city,day+1) for city,name in enumerate(flights[current]) if name)

        return dfs(0,-1)
