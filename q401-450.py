# Q418 Sentence Screen Fitting
class Solution:
    def wordsTyping(self, sentence, rows, cols):
        self.count = 0
        self.word_counter = 0
        self.words = len(sentence)
        for _ in range(rows):
            self.sequence = []
            self.recur(0,sentence[self.word_counter],cols)
        return self.count

    def recur(self,index,word,length):
        if index + len(word) <= length:
            self.sequence.extend(list(word))
            self.sequence.append('-')
            index += len(word) + 1

            if self.word_counter == self.words - 1:
                self.count += 1

            self.word_counter = (self.word_counter + 1) % self.words
            self.recur(index,sentence[self.word_counter],length)

# Q428 Serialize and Deserialize N-ary Tree
from collections import defaultdict

class Node:
    def __init__(self, val, children):
        self.val = val
        self.children = children

class Codec:

    def serialize(self, root):
        def dfs(node, vals):
            if not node:
                return
            vals.append(str(node.val))
            for child in node.children:
                dfs(child, vals)
            vals.append("#")

        vals = []
        dfs(root, vals)
        return " ".join(vals)


    def deserialize(self, data):
        def isplit(source, sep):
            sepsize = len(sep)
            start = 0
            while True:
                idx = source.find(sep, start)
                if idx == -1:
                    yield source[start:]
                    return
                yield source[start:idx]
                start = idx + sepsize

        def dfs(vals):
            val = next(vals)
            if val == "#":
                return None
            root = Node(int(val), [])
            child = dfs(vals)
            while child:
                root.children.append(child)
                child = dfs(vals)
            return root

        if not data:
            return None

        return dfs(iter(isplit(data, ' ')))
