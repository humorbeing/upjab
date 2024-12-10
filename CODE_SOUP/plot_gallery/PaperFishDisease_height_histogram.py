# import random
# random.seed(0)
# import shutil
# from scipy.stats import skew
from scipy.stats import skewnorm
import numpy as np
import matplotlib.pyplot as plt


sample_size = 4248357


hw_sample_mean = 0.9859418601170674
hw_sample_std = 0.6687966031141714
hw_sample_skewness = 236.37059671696136

h_sample_mean = 88.50360692380607
h_sample_std = 55.86775674928518
h_sample_skewness = 2.264709554319458

hw_ratios = skewnorm.rvs(
    hw_sample_skewness,
    loc=hw_sample_mean,
    scale=hw_sample_std,
    size=sample_size
    ).astype(np.float64)

heights = skewnorm.rvs(
    h_sample_skewness,
    loc=h_sample_mean,
    scale=h_sample_std,
    size=sample_size
    ).astype(np.int32)

# cut
# indx = hw_ratios < 0.25
# indx = (hw_ratios < 2) | (hw_ratios > 5)
indx = (hw_ratios > 0.4) & (hw_ratios < 1.3)

heights = heights[indx]

fig=plt.figure(figsize=(6.8,5))
ax1 = fig.add_subplot(111)

ax1.hist(heights, bins=100, label='Cropped Fish Images(Task)')
ax1.set_xlabel('Size (Number of Pixel) of Height')
ax1.set_ylabel('Count')
ax1.legend(loc='upper right')
ax1.set_title('Histogram of Size of Height')
# ax1.grid(True)
plt.show()

print('done')

