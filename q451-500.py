# Q457 Circular Array Loop
class Solution:
    #to find loops, just use one fast and one slow
    #Each time a possible attempt failed we mark every index on the path by zero, since zero is guaranteed to fail.
    #Problem asks only forward of backward solution we simply run it for positive indices and negative indices twice.
    def circularArrayLoop(self, nums):
        def next_index(nums, i):
            return (i + nums[i]) % len(nums)

        for i in range(len(nums)):
            if nums[i] == 0:
                continue

            slow = fast = i
            #checking positive or negative
            #if all are positive, product returns positive. Same for if all are negative
            while nums[next_index(nums, slow)] * nums[i] > 0 and nums[next_index(nums, fast)] * nums[i] > 0 and nums[next_index(nums, next_index(nums, fast))] * nums[i] > 0:
                slow = next_index(nums, slow)
                fast = next_index(nums, next_index(nums, fast))
                if slow == fast:
                    #check for loop with only one element
                    if slow == next_index(nums, slow):
                        break
                    return True

            #loop not found, set all element along the way to 0
            #this is an improvement. can work without.
            slow, val = i, nums[i]
            while nums[slow] * val > 0:
                tmp = next_index(nums, slow)
                nums[slow] = 0
                slow = tmp

        return False

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

# Q475 Heaters
class Solution:
    def findRadius(self, houses, heaters):
        houses.sort()
        heaters.sort()
        heaters=[float('-inf')]+heaters+[float('inf')] # add 2 fake heaters
        ans,i = 0,0
        for house in houses:
            while house > heaters[i+1]:  # search to put house between heaters
                i +=1
            dis = min (house - heaters[i], heaters[i+1]- house)
            ans = max(ans, dis)
        return ans

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
            count += 1
            i -= 1
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
                    #move() method automatically moves you, so you have to go back to original location to continue turning.
                    goBack(robot)
                robot.turnRight()
                d = (d+1) % len(directions)

        dfs((0, 0), robot, 0, set())

# Q497 Random Point in Non-overlapping Rectangles
class Solution:

    def __init__(self, rects):
        self.rects, self.ranges, sm = rects, [0], 0
        for x1, y1, x2, y2 in rects:
            sm += (x2 - x1 + 1) * (y2 - y1 + 1)
            self.ranges.append(sm)

    def pick(self):
        n = random.randint(0, self.ranges[-1] - 1)
        i = bisect.bisect(self.ranges, n)
        x1, y1, x2, y2 = self.rects[i - 1]
        n -= self.ranges[i - 1]
        return [x1 + n % (x2 - x1 + 1), y1 + n // (x2 - x1 + 1)]
