# Q457 Circular Array Loop


# Q465 Optimal Account Balancing
import collections

class Solution:
    def minTransfers(self, transactions):
        lookup = collections.defaultdict(int)
        for i in transactions:
            lookup[i[0]] += i[2]
            lookup[i[1]] += -i[2]
            if lookup[i[0]] == 0:
                del lookup[i[0]]

        def recur(lookup):
            minAcc = min(lookup,key=lookup.get)
            maxAcc = max(lookup,key=lookup.get)

            if lookup[maxAcc] - abs(lookup[minAcc]) > 0:
                print('Persion #{} gave person #{} ${}'.format(minAcc,maxAcc,abs(lookup[minAcc])))
                lookup[maxAcc] -= abs(lookup[minAcc])
                del lookup[minAcc]
            elif lookup[maxAcc] - abs(lookup[minAcc]) < 0:
                print('Persion #{} gave person #{} ${}'.format(minAcc,maxAcc,abs(lookup[maxAcc])))
                lookup[minAcc] += lookup[maxAcc]
                del lookup[maxAcc]
            else:
                print('Persion #{} gave person #{} ${}'.format(minAcc,maxAcc,abs(lookup[maxAcc])))
                del lookup[maxAcc]
                del lookup[minAcc]

        while lookup:
            recur(lookup)

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

        def dfs(pos, robot, lookup):
            if pos in lookup:
                return
            lookup.add(pos)

            robot.clean()
            for i in range(len(directions)):
                if robot.move():
                    dfs((pos[0]+directions[i][0],
                         pos[1]+directions[i][1]),
                        robot, lookup)
                    goBack(robot)
                robot.turnRight()

        dfs((0, 0), robot, 0, set())
