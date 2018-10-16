# 681 Next Closest Time
class Solution:
class Solution:
    def nextClosestTime(self, time):
        #cannot do [::-1] or anything of that sort in set(). It is not suscriptable
        time = [int(i) for i in time if i != ':']

        for j in range(time[-1]+1,10):
            if j in time and j != time[-1]:
                time[-1] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

        for j in range(time[-2]+1,6):
            if j in time and j != time[-2]:
                time[-2] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

        for j in range(time[-3]+1,5):
            if j in time and j != time[-3]:
                time[-3] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

        for j in range(time[-4]+1,3):
            if j in time and j != time[-4]:
                time[-4] = j
                time = ''.join(str(i) for i in time)
                return time[:2]+':'+time[2:]

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
            #automatic packing if multiple values assigned to one variable
            #second forloop done first
            if all(digit in allowed for block in divmod(cur, 60) for digit in divmod(block, 10)):
                #2 puts two spaces of padding minimum and 0 puts a 0 in front of the number if only 1 is present
                #d is just for signed integer decimal
                return "{:02d}:{:02d}".format(*divmod(cur, 60))

# 682


# 683 K Empty Slots
import bisect

class Solution:
    def kEmptySlots(self, flowers, k):
        days = [0] * len(flowers)
        for day, position in enumerate(flowers, 1):
            days[position - 1] = day

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

# Q686 Repeated String Match
class Solution:
    def repeatedStringMatch(self, A, B):
        if len(B) < len(A):
            return -1

        q = (len(B)-1) // len(A)+1
        for i in range(2):
            if B in A * (q+i):
                return q+i
        return -1
