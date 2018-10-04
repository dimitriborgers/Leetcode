# 681 Next Closest Time
class Solution:
    def nextClosestTime(self, time):
        time = [int(i) for i in time if i != ':']
        seq = set(time)
        #cannot do [::-1] or anything of that sort in set(). It is not suscriptable
        seq.remove(time[-1])
        for j in range(time[-1]+1,10):
            if j in seq:
                time[-1] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]
        seq.add(time[-1])

        seq.remove(time[-2])
        for j in range(time[-2]+1,6):
            if j in seq:
                time[-2] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]
        seq.add(time[-2])

        seq.remove(time[-3])
        for j in range(time[-3]+1,5):
            if j in seq:
                time[-3] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]
        seq.add(time[-3])

        seq.remove(time[-4])
        for j in range(time[-4]+1,3):
            if j in seq:
                time[-4] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]
        seq.add(time[-4])

        minimum = min(time)
        time = [minimum]*4
        time = ''.join(str(i) for i in time)
        #concatenation
        return time[:2]+':'+time[2:]

    def nextClosestTime2(self, time):
        cur = 60 * int(time[:2]) + int(time[3:])
        allowed = {int(x) for x in time if x != ':'}
        while True:
            cur = (cur + 1) % (24 * 60)
            if all(digit in allowed
                    for block in divmod(cur, 60)
                    for digit in divmod(block, 10)):
                return "{:02d}:{:02d}".format(*divmod(cur, 60))

# 682


# 683 K Empty Slots
import bisect

class Solution:
    def kEmptySlots(self, flowers, k):
        days = [0] * len(flowers)
        for day, position in enumerate(flowers, 1):
            days[position - 1] = day
        print(days)

        ans = float('inf')
        left, right = 0, k+1
        while right < len(days):
            for i in range(left + 1, right):
                if days[i] < days[left] or days[i] < days[right]:
                    left, right = i, i+k+1
                    break
            #else statement will occur after end of loop. Will not occur if break statement is encountered.
            else:
                ans = min(ans, max(days[left], days[right]))
                left, right = right, right+k+1

        return ans if ans < float('inf') else -1

    def kEmptySlots2(self, flowers, k):
        active = []
        #second arg in enumerate lets you choose what number to start enumeration on.
        for day, flower in enumerate(flowers, 1):
            #bisect.bisect tells you where exactly the value would be inserted
            i = bisect.bisect(active, flower)
            #i-(i>0) -> i>0 is 1 if true, 0 if not
            for neighbor in active[i-(i>0):i+1]:
                if abs(neighbor - flower) - 1 == k:
                    return day
            #bisect.insort knows where to insert element
            bisect.insort(active,flower)
        return -1
