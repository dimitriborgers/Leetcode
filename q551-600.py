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
