# Q482 License Key Formatting
class Solution:
    def licenseKeyFormatting(self, S, K):
        S = [i.upper() for i in S if i.isalnum()]
        i = len(S)-1
        count = 0
        while i > 0:
            if count == K-1:
                S.insert(i,'-')
                count = -1
            count+=1
            i-=1
        return ''.join(S)

# Q489 Robot Room Cleaner
class Solution:
    def cleanRoom(self, robot):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def goBack(robot):
            robot.turnLeft()
            robot.turnLeft()
            robot.move()
            robot.turnRight()
            robot.turnRight()

        def dfs(pos, robot, d, lookup):
            if pos in lookup:
                return
            lookup.add(pos)

            robot.clean()
            for _ in directions:
                if robot.move():
                    dfs((pos[0]+directions[d][0],
                         pos[1]+directions[d][1]),
                        robot, d, lookup)
                    goBack(robot)
                robot.turnRight()
                d = (d+1) % len(directions)

        dfs((0, 0), robot, 0, set())
