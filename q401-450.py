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
class Node:
    def __init__(self, val, children):
        self.val = val
        self.children = children

class Codec:

    #store an ‘end of children’ marker with every node.
    def serialize(self, root):
        def dfs(node):
            if not node:
                return
            vals.append(str(node.val))
            for child in node.children:
                dfs(child)
            vals.append("#")

        vals = []
        dfs(root)
        return " ".join(vals)


    def deserialize(self, data):
        #split is not actually needed
        #to not use it, remove the ' ' (space) in between the numbers in the serialized data, and then just call on: return dfs(iter(data))
        def isplit(source, sep):
            sepsize = len(sep)
            start = 0
            while True:
                idx = source.find(sep, start)
                #find returns -1 if not found
                if idx == -1:
                    yield source[start:]
                    #return statement acts as a stop iteration
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

        #iter creates an iterator from iterable
        #iter can make an iterator from a string
        return dfs(iter(isplit(data, ' ')))
