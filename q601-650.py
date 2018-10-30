#642 Design Search Autocomplete System
import collections
import bisect

class TrieNode:

    def __init__(self):
        #A simple trie tree only needs self.leaves to keep track
        self.__TOP_COUNT = 3
        self.infos = []
        self.leaves = {}

    def insert(self, s, times):
        #self represents the current instance of the object TrieNode()
        cur = self
        cur.add_info(s, times)
        for c in s:
            if c not in cur.leaves:
                cur.leaves[c] = TrieNode()
            cur = cur.leaves[c]
            cur.add_info(s, times)

    def add_info(self, s, times):
        for p in self.infos:
            if p[1] == s:
                p[0] = -times
                break
        else:
            self.infos.append([-times, s])
        self.infos.sort()
        if len(self.infos) > self.__TOP_COUNT:
            self.infos.pop()

class AutocompleteSystem:

    def __init__(self, sentences, times):
        #can't do print(instanceName.__trie since name mangling with __)
        self.__trie = TrieNode()
        self.__cur_node = self.__trie
        self.__search = []
        self.__sentence_to_count = collections.defaultdict(int)
        for sentence, count in zip(sentences, times):
            self.__sentence_to_count[sentence] = count
            self.__trie.insert(sentence, count)

    def input(self, c):
        result = []
        if c == '#':
            self.__sentence_to_count["".join(self.__search)] += 1
            self.__trie.insert("".join(self.__search), self.__sentence_to_count["".join(self.__search)])
            self.__cur_node = self.__trie
            self.__search = []
        else:
            self.__search.append(c)
            #This check makes sure you stop comparing onces one letter is not in tree
            if self.__cur_node:
                if c not in self.__cur_node.leaves:
                    self.__cur_node = None
                    return []
                self.__cur_node = self.__cur_node.leaves[c]
                result = [p[1] for p in self.__cur_node.infos]
        return result
