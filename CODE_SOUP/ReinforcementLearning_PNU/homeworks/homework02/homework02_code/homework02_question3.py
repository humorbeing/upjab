import numpy as np
import matplotlib.pyplot as plt

with open('ex1data1.txt', 'r') as f:
    lines = f.readlines()

x = []
y = []
for line in lines:
    line1 = line.strip('\n')
    f1, f2 = line1.split(',')
    x.append(float(f1))
    y.append(float(f2))


x = np.array(x)
y = np.array(y)
y = y[:, None]

A = np.ones((len(x), 2), dtype=float)
A[:, 1] = x

AT = A.T
AT_A = np.matmul(AT, A)
AT_A_inv = np.linalg.inv(AT_A)

ATA1AT = np.matmul(AT_A_inv, AT)
theta = np.matmul(ATA1AT, y)

xlist = np.linspace(0, 25, 1000)
newA = np.ones((len(xlist), 2), dtype=float)
newA[:, 1] = xlist
ylist = np.matmul(newA, theta)

plt.plot(xlist, ylist, 'r--', alpha=0.8)
plt.scatter(x, y, marker='.', color='k')
plt.savefig('homework02_question3.png')
plt.show()
