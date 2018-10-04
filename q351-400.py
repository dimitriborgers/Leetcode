# Q399 Evaluate Division
import collections

class Solution:
    def calcEquation(self, equations, values, query):
        def check(up, down, lookup, visited):
            if up in lookup and down in lookup[up]:
                return (True, lookup[up][down])
            for k, v in lookup[up].items():
                if k not in visited:
                    visited.add(k)
                    tmp = check(k, down, lookup, visited)
                    if tmp[0]:
                        return (True, v * tmp[1])
            return (False, 0)

        lookup = collections.defaultdict(dict)
        for i, e in enumerate(equations):
            lookup[e[0]][e[1]] = values[i]
            if values[i]:
                lookup[e[1]][e[0]] = 1 / values[i]

        result = []
        for q in query:
            visited = set()
            tmp = check(q[0], q[1], lookup, visited)
            result.append(tmp[1] if tmp[0] else -1)
        return result
