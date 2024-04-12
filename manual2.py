import itertools

items = ["pizza", "wasabi", "snow", "seashells"]
table = [[1, 0.48, 1.52, 0.71],
         [2.05, 1, 3.26, 1.56],
         [0.64, 0.3, 1, 0.46],
         [1.41, 0.61, 2.08, 1]]

start = 0 #seashells
end = 0 #seashells
permutations = list(itertools.permutations([0, 1, 2, 3])) #.append(list(itertools.permutations(1, 2, 3, 3)))
per2 = list(itertools.permutations([1, 1, 2, 0]))
per3 = list(itertools.permutations([1, 2, 2, 0]))
per4 = list(itertools.permutations([1, 2, 0, 0]))
per5 = list(itertools.permutations([1, 1, 2, 2]))
per6 = list(itertools.permutations([2, 2, 0, 0]))
per7 = list(itertools.permutations([1, 1, 0, 0]))
# perlist = [permutations, per2, per3, per4, per5, per6, per7]
perlist = [per7]
# per3 = list(itertools.permutations([0, 1, 2, 3]))


# for i in range(len(permutations)):
#     permutations[i] = [0].append(permutations[i].append([0]))

for permutation in perlist:
    returnlist = []
    for entry in permutation:
        print(f"{items[3]} >> {items[entry[0]]} >> {items[entry[1]]} >> {items[entry[2]]} >> {items[entry[3]]} >> {items[3]}")
        ans = 1
        ans *= table[3][entry[0]]
        ans *= table[entry[0]][entry[1]]
        ans *= table[entry[1]][entry[2]]
        ans *= table[entry[2]][entry[3]]
        ans *= table[entry[3]][3]
        # ans *= table[entry[j]][entry[j + 1]]
        print(ans)
        returnlist.append(ans)
    # print(list(zip(permutations, returnlist)))
    print(max(returnlist))

