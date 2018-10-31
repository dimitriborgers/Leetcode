# Q731 My Calendar II
from collections import defaultdict

class MyCalendarTwo:

    def __init__(self):
        self.calendar = defaultdict(int)

    def book(self, start, end):
        #The loop will not start if self.calendar is empty
        for k,v in self.calendar.items():
            if (k[0] <= start < k[1]) or (k[1] > end > k[0]):
                if self.calendar[k] == 2:
                    return False
                self.calendar[k] += 1
        self.calendar[(start,end)] = 1
        return True

#Q 734 Sentence Similarity
import collections

class Solution:
    def areSentencesSimilar(self, words1, words2, pairs):
        words1 = words1.split(' ')
        words2 = words2.split(' ')

        if len(words1) != len(words2):
            return False

        lookup = collections.defaultdict(list)
        for left,right in pairs:
            lookup[left].append(right)
            lookup[left].append(left)
            lookup[right].append(left)
            lookup[right].append(right)

        #more pythonic way of checking something for every element in list
        return all(word1 in lookup[word2] and word2 in lookup[word1] for word1,word2 in zip(words1,words2))

class Solution2:
    def areSentencesSimilar(self, words1, words2, pairs):
        if len(words1) != len(words2):
            return False

        lookup = set(tuple(i) for i in pairs)
        return all(w1 == w2 or (w1, w2) in lookup or (w2, w1) in lookup for w1, w2 in zip(words1, words2))
