import math
import itertools

def has_arbitrage(M):
    
    n = len(M)

    # Apply log on edges, flip signs
    for i, j in itertools.product(range(n), repeat=2):
        M[i][j] = -math.log2(M[i][j])

    # Track shortest "dist" from source for each node
    D = [math.inf for _ in range(n)]
    D[0] = 0
 
    # Run Bellman-Ford
    for _ in range(n):
        for i, j in itertools.product(range(n), repeat=2):
            D[j] = min(D[j], D[i] + M[i][j])
    
    # Check for -ve cycles
    for i, j in itertools.product(range(n), repeat=2):
        if D[i] + M[i][j] < D[j]:
            return True

    return False

table = [[1, 0.48, 1.52, 0.71],
         [2.05, 1, 3.26, 1.56],
         [0.64, 0.3, 1, 0.46],
         [1.41, 0.61, 2.08, 1]]

print(has_arbitrage(table))
