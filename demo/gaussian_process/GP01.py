import numpy as np


class Squared_Exponential_Kernel:
    def __init__(self, length_scale=1.0, sigma=1.):
        self.length_scale = length_scale
        self.sigma_f = sigma
    
    def get(self, a, b):
        temp11 = (a - b)**2
        temp12 = temp11 / (2 * (self.length_scale**2))
        temp13 = np.exp(-temp12) * (self.sigma_f ** 2)
        return temp13
    

    def check_dimension(self, list_a):
        # shape (n, 1): to be exact when do matrix multiplication
        arr = np.array(list_a)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        elif arr.ndim == 1:
            return arr.reshape(-1, 1)
        elif arr.ndim == 2 and arr.shape[1] == 1:
            return arr
        else:
            raise ValueError("Input must be a scalar, a 1D list/array, or a 2D array with shape (n, 1)")

    def __call__(self, list_a, list_b):
        list_a = self.check_dimension(list_a)
        list_b = self.check_dimension(list_b)
        n1 = len(list_a)
        n2 = len(list_b)
        re = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                re[i, j] = self.get(list_a[i].item(), list_b[j].item())
        return re





class Gaussian_Process_Regression:
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.Kxx = self.kernel(x, x)
        self.Kxx_inv = np.linalg.inv(self.Kxx + 1e-8 * np.eye(len(x)))

    def predict(self, x_star):
        K_x_star = self.kernel(self.x, x_star)
        K_star_star = self.kernel(x_star, x_star)

        temp11 = K_x_star.T @ self.Kxx_inv

        mu_star = temp11 @ self.y
        SIGMA2_star = K_star_star - temp11 @ K_x_star
        SIGMA_star = np.sqrt(np.diag(SIGMA2_star))

        return {
            'mu_star': mu_star,
            'SIGMA2_star': SIGMA2_star,
            'SIGMA_star': SIGMA_star
        }
        
        # return mu_star, SIGMA2_star


x = np.array((1,2,3,4)).reshape(-1, 1)  # shape (n, 1): to be exact when do matrix multiplication
y = np.array((1,1.5,3,2)).reshape(-1, 1)




kernel = Squared_Exponential_Kernel(length_scale=1.0, sigma=1.)
model = Gaussian_Process_Regression(kernel)

model.fit(x, y)



from matplotlib import pyplot as plt


plt.figure(figsize=(8, 6))
plt.xlim(-0.1, 5.1)
plt.ylim(-2, 7)
plt.scatter(x, y)
plt.show()


x_star = np.array((1.5)).reshape(-1, 1)

result = model.predict(x_star)
mu_star = result['mu_star']
SIGMA2_star = result['SIGMA2_star']
SIGMA_star = result['SIGMA_star']

plt.figure(figsize=(8, 6))
plt.xlim(-0.1, 5.1)
plt.ylim(-2, 7)
plt.scatter(x, y)
plt.scatter(x_star, mu_star, color='r')
# plt.errorbar(x_star, mu_star, yerr=np.sqrt(np.diag(SIGMA2_star)), fmt='o', color='r', capsize=5)
plt.errorbar(x_star, mu_star, yerr=SIGMA_star, fmt='o', color='r', capsize=5, alpha=0.3)
plt.errorbar(x_star, mu_star, yerr=SIGMA_star*2, fmt='o', color='r', capsize=5, alpha=0.2)
plt.errorbar(x_star, mu_star, yerr=SIGMA_star*3, fmt='o', color='r', capsize=5, alpha=0.1)

plt.show()




x_star = np.array((1.5)).reshape(-1, 1)

result = model.predict(x_star)
mu_star = result['mu_star']
SIGMA2_star = result['SIGMA2_star']
SIGMA_star = result['SIGMA_star']

plt.figure(figsize=(8, 6))
plt.xlim(-0.1, 5.1)
plt.ylim(-2, 7)
plt.scatter(x, y)
plt.scatter(x_star, mu_star, color='r')
# plt.errorbar(x_star, mu_star, yerr=np.sqrt(np.diag(SIGMA2_star)), fmt='o', color='r', capsize=5)
plt.errorbar(x_star, mu_star, yerr=SIGMA_star, fmt='o', color='r', capsize=5, alpha=0.3)
plt.errorbar(x_star, mu_star, yerr=SIGMA_star*2, fmt='o', color='r', capsize=5, alpha=0.2)
plt.errorbar(x_star, mu_star, yerr=SIGMA_star*3, fmt='o', color='r', capsize=5, alpha=0.1)


num_star = 100
x_star = np.linspace(0, 5, num_star).reshape(-1,1)



result = model.predict(x_star)
mu_star = result['mu_star']
SIGMA2_star = result['SIGMA2_star']
SIGMA_star = result['SIGMA_star']




# plt.scatter(x_star, mu_star, color='r')
plt.plot(x_star, mu_star, 'r-', label='Predictive mean')


x_star_flat = x_star.reshape(-1).flat
y_star_plot = mu_star.reshape(-1)
std_y_star = SIGMA_star.reshape(-1)
plt.gca().fill_between(x_star_flat, y_star_plot - std_y_star * 1, y_star_plot + std_y_star * 1, color='k', alpha=0.3)
plt.gca().fill_between(x_star_flat, y_star_plot - std_y_star * 2, y_star_plot + std_y_star * 2, color='k', alpha=0.2)
plt.gca().fill_between(x_star_flat, y_star_plot - std_y_star * 3, y_star_plot + std_y_star * 3, color='k', alpha=0.1)



plt.show()

print('done')