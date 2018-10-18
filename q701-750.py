# Q731 My Calendar II
from collections import defaultdict

class MyCalendarTwo:

    def __init__(self):
        self.calendar = defaultdict(int)

    def book(self, start, end):
        if not self.calendar:
            self.calendar[(start,end)] = 1

        for k,v in self.calendar.items():
            if (k[0] <= start < k[1]) or (k[1] > end > k[0]):
                if self.calendar[k] == 2:
                    return False
                self.calendar[k] += 1
        self.calendar[(start,end)] = 1
        return True
