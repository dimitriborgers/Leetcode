# Q393 UTF-8 Validation
#0b is like 0x - it indicates the number is formatted in binary (0x indicates the number is in hex).
#bin(30)[2:].zfill(8) //if bin(30) is less than 8 digits, it fills the left side with 0s.
#bin() // returns a string
#print(1 == 0b1) //returns True
#print('{:08b}'.format(2)) //adds 0s to the left of digit inputted
#print('{:8b}'.format(2)) //adds space to the left of digit inputted
#print('{:8b}'.format(2)) will output 2 in binary
#print('{:8d}'.format(2)) will output 2 in decimal
#bin(128) //10000000 (1 + 7*0)
#bin(255) //11111111 (8*1)
#int(binary_number,2) //converts binary to decimal
#print(int(bin(a)[a.bit_length()-6:],2))  -> get 8 least sig bits
#bit_length() does not include 0b
class Solution:
    def validUtf8(self, data):
        count = 0
        for c in data:
            if count == 0:
                if (c >> 5) == 0b110:
                    count = 1
                elif (c >> 4) == 0b1110:
                    count = 2
                elif (c >> 3) == 0b11110:
                    count = 3
                elif (c >> 7):
                    return False
            else:
                if (c >> 6) != 0b10:
                    return False
                count -= 1
        return count == 0

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
                    #this check of tmp[0] prevents the function from returning too quickly. It allows all steps in loop
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
            #can put a conditional statement in an append() statement
            result.append(tmp[1] if tmp[0] else -1)
        return result
