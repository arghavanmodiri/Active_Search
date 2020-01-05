from pyGPs import cov


r = cov.RBF(1.0986, 1.0986)
p = [[1, 3],[2, 1],[2, 2],[1, 1]]

print(r.getCovMatrix(x=p,mode='train'))