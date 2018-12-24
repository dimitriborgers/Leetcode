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
# Time limit exceeded
class Solution1:
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

# Uses dfs and memoization
class Solution2:
    def maxVacationDays(self, flights, days):
        n = len(flights)
        k = len(days[0])
        graph = collections.defaultdict(list)
        for i in range(n):
            for j in range(n):
                if flights[i][j]:
                    graph[j].append(i)
            graph[i].append(i)
        dp = [[-float('inf')]*n for i in range(k+1)]
        dp[0][0] = 0
        for i in range(1,k+1):
            for j in range(n):
                for k in graph[j]:
                    dp[i][j] = max(dp[i][j],dp[i-1][k])
                dp[i][j] += days[j][i-1]
        return max(dp[-1])

# Q579 Find Cumulative Salary of an Employee
SELECT
    E1.id,
    E1.month,
    (IFNULL(E1.salary, 0) + IFNULL(E2.salary, 0) + IFNULL(E3.salary, 0)) AS Salary
FROM
    (SELECT
        id, MAX(month) AS month
    FROM
        Employee
    GROUP BY id
    HAVING COUNT(*) > 1) AS maxmonth
        LEFT JOIN
    Employee E1 ON (maxmonth.id = E1.id
        AND maxmonth.month > E1.month)
        LEFT JOIN
    Employee E2 ON (E2.id = E1.id
        AND E2.month = E1.month - 1)
        LEFT JOIN
    Employee E3 ON (E3.id = E1.id
        AND E3.month = E1.month - 2)
ORDER BY id ASC , month DESC
;
