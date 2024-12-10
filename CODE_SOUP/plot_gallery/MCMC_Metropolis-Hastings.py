import numpy as np
import matplotlib.pyplot as plt

def p_star1(x):
    p_ = np.exp(0.5 * (x-6.2)**2 - 0.1*(x-6)**4)
    return p_

xs = np.linspace(2, 10, 10000)
ys = p_star1(xs)

plt.plot(xs, ys)
# plt.savefig('p1_Target.jpg')
plt.show()

NUM_SAMPLE = 100000
BURNIN = 1000
THINNING = 3
SIGMA = 0.1


sample_x = 0
samples = []
samples.append(p_star1(sample_x))
xs = []
xs.append(sample_x)
while True:
    print(f'{len(samples)=}')
    if len(samples) == NUM_SAMPLE:
        break
    
    x_old = sample_x
    x_new = np.random.normal(x_old, SIGMA, 1).item()

    p_old = p_star1(x_old)
    p_new = p_star1(x_new)
    r = p_new / p_old
    if r >= 1:
        samples.append(p_new)
        sample_x = x_new
        xs.append(sample_x)
    elif r > np.random.rand():
            samples.append(p_new)
            sample_x = x_new
            xs.append(sample_x)


burnin_samples = samples[BURNIN:]
burnin_thinning_samples = burnin_samples[::THINNING]


s = burnin_thinning_samples
s_mean = np.mean(s)
print(f'{s_mean=}')

burnin_xs = xs[BURNIN:]
burnin_thinning_xs = burnin_xs[::THINNING]
x = burnin_thinning_xs



plt.hist(x, bins=100)
# plt.savefig('p1_Histogram.jpg')
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm

def p_star2(x, y):
    p_ = np.exp(-x**2 + 26.6*x -y**2+26.6*y-1.8*x*y-186.2) \
        + np.exp(-x**2 + x -y**2 + y + 1.8*x*y -5)
    return p_

# p_star2
x = np.linspace(-2, 13, 200)
y = np.linspace(-2, 13, 200)
xs, ys = np.meshgrid(x, y)
zs = p_star2(xs, ys)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set(xlim=(-2.5, 13.5), ylim=(-2.5, 13.5), zlim=(-1, 1.4),
       xlabel='X', ylabel='Y', zlabel='Z')

ax = fig.add_subplot(1, 2, 2, projection='3d')
# ax = plt.figure().add_subplot(projection='3d')
# Plot the 3D surface
ax.plot_surface(xs, ys, zs, edgecolor='royalblue', lw=0.3, rstride=8, cstride=8,
                alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph.
ax.contour(xs, ys, zs, zdir='z', offset=-1, cmap='coolwarm')
ax.contour(xs, ys, zs, zdir='x', offset=-2.5, cmap='coolwarm')
ax.contour(xs, ys, zs, zdir='y', offset=13.5, cmap='coolwarm')

ax.set(xlim=(-2.5, 13.5), ylim=(-2.5, 13.5), zlim=(-1, 1.4),
       xlabel='X', ylabel='Y', zlabel='Z')

# plt.savefig('p2_Target.jpg')
plt.show()

NUM_SAMPLE = 500000
BURNIN = 1000
THINNING = 3
SIGMA = 0.1



sample_x = 0
sample_y = 0
xys = []

xys.append([sample_x, sample_y])

samples = []
samples.append(p_star2(sample_x, sample_y))


while True:
    print(f'{len(samples)=}')
    if len(samples) == NUM_SAMPLE:
        break
    
    x_old = sample_x
    x_new = np.random.normal(x_old, SIGMA, 1).item()

    y_old = sample_y
    y_new = np.random.normal(y_old, SIGMA, 1).item()

    p_old = p_star2(x_old, y_old)
    p_new = p_star2(x_new, y_new)
    r = p_new / p_old
    if r >= 1:
        samples.append(p_new)
        sample_x = x_new        
        sample_y = y_new
        xys.append([sample_x,sample_y])
        
    elif r > np.random.rand():
            samples.append(p_new)
            sample_x = x_new        
            sample_y = y_new
            xys.append([sample_x,sample_y])
        

burnin_samples = samples[BURNIN:]
burnin_thinning_samples = burnin_samples[::THINNING]



s = burnin_thinning_samples
s_mean = np.mean(s)
print(f'{s_mean=}')

burnin_xys = xys[BURNIN:]
burnin_thinning_xys = burnin_xys[::THINNING]
xy = burnin_thinning_xys

x = []
y = []
for i, j in xy:
    x.append(i)
    y.append(j)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(x, y, bins=(100,100))
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
plt.xlabel ("X")
plt.ylabel ("Y")

# plt.savefig('p2_Histogram.jpg')
plt.show()
# print('done')