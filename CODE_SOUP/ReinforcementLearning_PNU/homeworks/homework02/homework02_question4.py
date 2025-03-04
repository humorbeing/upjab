import numpy as np
import matplotlib.pyplot as plt
import random

with open('ex1data1.txt', 'r') as f:
    lines = f.readlines()

x = []
y = []
for line in lines:
    line1 = line.strip('\n')
    f1, f2 = line1.split(',')
    x.append(float(f1))
    y.append(float(f2))


step_sizes = [
    0.1,
    0.01,
    0.001,
    0.0001,
    0.00001,
]
MAX_ITERATION = 5000

last_time_theta0 = 0
last_time_theta1 = 0
is_num0 = True
is_num1 = True

def line(x, t0, t1, plotting=False):
    global last_time_theta0
    global last_time_theta1
    global is_num0
    global is_num1
    if is_num0:
        if np.isinf(t0):
            is_num0 = False
        else:
            last_time_theta0 = t0
    
    if is_num1:
        if np.isinf(t1):
            is_num1 = False
        else:
            last_time_theta1 = t1
    
    if plotting:
        if np.isnan(t0):
            t0 = last_time_theta0
        if np.isnan(t1):
            t1 = last_time_theta1
    y_hat = t0 + t1 * x
    return y_hat

thetas = []
for step_size in step_sizes:
    theta0 = random.random()
    theta1 = random.random()

    for _ in range(MAX_ITERATION):
        accumulating_theta0 = 0
        accumulating_theta1 = 0
        for i in range(len(x)):
            accumulating_theta0 = accumulating_theta0 + (line(x[i], theta0, theta1) - y[i])
            accumulating_theta1 = accumulating_theta1 + (line(x[i], theta0, theta1) - y[i]) * x[i]

        temp0 = theta0 - step_size * accumulating_theta0 / len(x)
        temp1 = theta1 - step_size * accumulating_theta1 / len(x)

        theta0 = temp0
        theta1 = temp1
        print(f'theta0: {theta0:0.08f}, theta1: {theta1:0.08f}')
    thetas.append([theta0, theta1])


xlist = np.linspace(-5, 30, 1000)
colors = ['r','g','b','c', 'k']

if is_num0:
    msg = ''
else:
    msg = ' (Fail to converge)'    

labels = [
    f'0.1{msg}',
    '0.01',
    '0.001',
    '0.0001',
    '0.00001',
]
for i in range(len(thetas)):
    theta = thetas[i]
    ylist = []
    for xi in xlist:
        y_hat = line(xi, theta[0], theta[1], plotting=True)
        ylist.append(y_hat)
    ylist = np.array(ylist)
    plt.plot(xlist, ylist, f'{colors[i]}--', linewidth=2, alpha=0.8, label=labels[i])


plt.scatter(x, y, marker='.', color='k')
plt.axis([-2, 27, -7, 27])
plt.legend(loc='lower right')
plt.savefig('homework02_question4.png')
plt.show()


