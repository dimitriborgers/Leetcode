# Q539 Minimum Time Difference
class Solution:
    def findMinDifference(self, timePoints):
        minutes = [int(x[:2]) * 60 + int(x[3:]) for x in timePoints]
        minutes.sort()
        #modulo returns positive if negative % positive
        #ie. -1249 % 1440 = 191
        return min((y - x) % (24 * 60) for x, y in zip(minutes, minutes[1:] + minutes[:1]))
