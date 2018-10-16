# Q271 Encode and Decode Strings
class Codec:

    def encode(self, strs):
        encoded_str = ''
        for s in strs:
            encoded_str +=  '0000000' + str(len(s)) + s
        return encoded_str

    def decode(self, s):
        i = 0
        strs = []
        while i < len(s):
            l = int(s[i+7])
            strs.append(s[i+8:i+8+l])
            i += 8+l
        return strs

# Q273. Integer to English Words
class Solution:
    def numberToWords(self, num):
        if num == 0:
            return "Zero"

        lookup = {0: "Zero", 1:"One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen", 16: "Sixteen", 17: "Seventeen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty", 30: "Thirty", 40: "Forty", 50: "Fifty", 60: "Sixty", 70: "Seventy", 80: "Eighty", 90: "Ninety"}

        unit = ["", "Thousand", "Million", "Billion"]

        res, i = [], 0
        while num:
            cur = num % 1000
            if num % 1000:
                res.append(self.threeDigits(cur, lookup, unit[i]))
            num //= 1000
            i += 1
        return " ".join(res[::-1])

    def threeDigits(self, num, lookup, unit):
        res = []
        if num // 100:
            res = [lookup[num // 100] + " " + "Hundred"]
        if num % 100:
            res.append(self.twoDigits(num % 100, lookup))
        if unit != "":
            res.append(unit)
        return " ".join(res)

    def twoDigits(self, num, lookup):
        if num in lookup:
            return lookup[num]
        return lookup[(num // 10) * 10] + " " + lookup[num % 10]

# Q285 Inorder Successor in BST
class Solution:
    def inorderSuccessor(self, root, p):
        self.p = p
        self.found = False

        def inorderTraversal(root):
            if root.left:
                inorderTraversal(root.left)
            if root.val == self.p:
                self.found = True
            #since it uses self.found, it acts like a global variable
            if self.found:
                return root.val
            if root.right:
                inorderTraversal(root.right)

        #this method can only be called after its definition. Python is interpreted, so it goes line by line.
        return inorderTraversal(root)

#Q 299 Bulls and Cows
class Solution:
    def getHint(self, secret, guess):
        used = []
        bulls = 0
        cows = 0
        for i in range(len(secret)):
            if secret[i] == guess[i]:
                bulls += 1
                used.append(i)
        for i in used:
            #str does not have pop() operation
            secret = secret[:i]+secret[i+1:]
            guess = guess[:i]+guess[i+1:]
        for i in secret:
            if i in guess:
                cows += 1
        return '{}A{}B'.format(bulls,cows)

