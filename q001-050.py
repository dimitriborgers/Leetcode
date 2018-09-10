# Q1 Two Sum

# Q2 Add Two Numbers
import linked_list as ll
class LinkedList:

    class ListNode:
        __slots__ = 'value','next'

        def __init__(self, _val, _next):
            self.value = _val
            self.next = _next

    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def __len__(self):
        return self.size

    def __repr__(self):
        if self.size == 0:
            return '[]'
        outcome = []
        current = self.head
        while current:
            outcome.append(current.value)
            current = current.next
        return "{}".format(outcome)

    def addHead(self,value):
        if self.head == None:
            self.head = self.tail = self.ListNode(value, self.head)
        else:
            self.head = self.ListNode(value, self.head)
        self.size += 1

    def addTail(self,value):
        if self.head == None:
            self.head = self.tail = self.ListNode(value, self.head)
        else:
            self.tail.next = self.ListNode(value,None)
            self.tail = self.tail.next
        self.size += 1

    def removeHead(self):
        old_head = self.head
        self.head = self.head.next
        return old_head

    def removeTail(self):
        current = self.head
        while current.next.next:
            current = current.next
        old_head = current.next
        self.tail = current
        self.tail.next = None
        return old_head

class Solution:

    def addTwoNumbers(self, l1, l2):
        l3 = LinkedList()
        remainder = 0
        overflow = 0
        while l1.head:
            temp1 = l1.removeHead()
            temp2 = l2.removeHead()

            if remainder == 0:
                if temp1.value+temp2.value > 9:
                    if not temp1.next:
                        overflow = 1
                    remainder = 1
                    addition = temp1.value+temp2.value-10
                else:
                    addition = temp1.value+temp2.value
            else:
                if temp1.value+temp2.value+remainder > 9:
                    if not temp1.next:
                        overflow = 1
                    remainder = 1
                    addition = temp1.value+temp2.value+remainder-10
                else:
                    addition = temp1.value+temp2.value+remainder

            l3.addTail(addition)
            if overflow == 1:
                l3.addTail(1)
            print(l3)

# Q3 Longest Substring Without Repeating Characters
class Solution:
    def lengthOfLongestSubstring(self, s):
        i = j = 0
        seq = set()
        maxCount = 0
        count = 0
        while i < len(s) and j < len(s):
            if s[j] not in seq:
                seq.add(s[j])
                j += 1
                count = len(seq)
                if count > maxCount:
                    maxCount = count
            else:
                seq.remove(s[i])
                i += 1
        print(maxCount)

# Q4 Median of Two Sorted Arrays
class Solution:
    def median(self, A, B):
    m, n = len(A), len(B)
    if m > n:
        A, B, m, n = B, A, n, m
    if n == 0:
        raise ValueError

    imin, imax, half_len = 0, m, (m + n + 1) / 2
    while imin <= imax:
        i = (imin + imax) / 2
        j = half_len - i
        if i < m and B[j-1] > A[i]:
            # i is too small, must increase it
            imin = i + 1
        elif i > 0 and A[i-1] > B[j]:
            # i is too big, must decrease it
            imax = i - 1
        else:
            # i is perfect

            if i == 0: max_of_left = B[j-1]
            elif j == 0: max_of_left = A[i-1]
            else: max_of_left = max(A[i-1], B[j-1])

            if (m + n) % 2 == 1:
                return max_of_left

            if i == m: min_of_right = B[j]
            elif j == n: min_of_right = A[i]
            else: min_of_right = min(A[i], B[j])

            return (max_of_left + min_of_right) / 2.0

# Q5 Longest Palindromic Substring


# Q6


# Q7


# Q8
