import itertools

items = ["seashells", "pizza", "wasabi", "snow"]
table = [[1, 0.48, 1.52, 0.71],
         [2.05, 1, 3.26, 1.56],
         [0.64, 0.3, 1, 0.46],
         [1.41, 0.61, 2.08, 1]]

start = 0 #seashells
end = 0 #seashells
permutations = list(itertools.permutations([0, 1, 2, 3])) #.append(list(itertools.permutations(1, 2, 3, 3)))
returnlist = []

# for i in range(len(permutations)):
#     permutations[i] = [0].append(permutations[i].append([0]))

for entry in permutations:
    ans = 1
    ans *= table[0][entry[1]]
    ans *= table[entry[1]][entry[2]]
    ans *= table[entry[2]][entry[3]]
    ans *= table[entry[3]][0]
    # ans *= table[entry[j]][entry[j + 1]]
    returnlist.append(ans)


print(list(zip(permutations, returnlist)))
print(max(returnlist))
