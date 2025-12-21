# Efficient GP


\subsubsection{Disagreement-based Acquisition} Query-by-Committee (QBC), choose new samples where a group of different models shows the greatest disagreement \cite{wang2025active}. In this approach, a surrogate ensemble—typically including Radial Basis Function Networks, Gaussian Process Regression, and Support Vector Regression—is used to identify points where their predictions differ the most. By focusing on regions with high disagreement, QBC helps address the bias–variance trade-off that often arises when fitting Gaussian Process models to small datasets \cite{riis2022bayesian}.

efficient gaussian process


from efficientGP.utils.common import params_count
params_count(gp_model.model)
    

effNo.', 'head', 'Q', 'rpm']


array([[-7.210603 ],
       [-7.6590548]], dtype=float32)
y_torque_pred
array([-3.79009414, -3.79009414])

# code for mean and kernel

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, output_dim=64):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.LinearMean(input_size=train_x.size(-1))
        # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
        #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
        #     num_dims=2, grid_size=100
        # )
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)) + gpytorch.kernels.ConstantKernel()
        self.feature_extractor = LargeFeatureExtractor(train_x.size(-1), output_dim=output_dim)

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        


# logs

output_name_list = [
    'Pt_in [Pa]',
    'Pt_out [Pa]',
    'del_Pt [Pa]',
    'Torque [N m]',
    'Efficiency [%]',
    'Efficiency (Pt_out - Pt_in)',
]

output_name_list = [
    'Pt_in [Pa]',
    'Pt_out [Pa]',
    
    'Torque [N m]',
    'Efficiency [%]',
    'Efficiency (Pt_out - Pt_in)',
]



metric_name_list = [
    'explained_variance_score',
    'r2_score',
    'd2_absolute_error_score',
    'd2_pinball_score',
    'd2_tweedie_score',
    'mean_absolute_error',
    'median_absolute_error',
    'mean_squared_error',
    'rmse',
    'mean_squared_log_error',
    'rmsle',
    'mean_absolute_percentage_error',
    'max_error',
    'mean_pinball_loss(alpha=0.5)',
    'mean_poisson_deviance',
    'mean_gamma_deviance',
    'mean_tweedie_deviance(power=0.0)',
]


# colors

#12436D
#005CAB
#28A197
#F46A25
#E31B23
#3D3D3D
#FFC325


#125A56
#00767B
#238F9D
#FFB954
#FD9A44
#F57634
#E94C1F
#D11807

# Idea

- train with the metric?
- normalization from training data




gpytorch GaussianLikelihood ConstantMean ScaleKernel(RBFKernel) ExactMarginalLogLikelihood



import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

x = torch.tensor(np.array((1,2,3,4)).reshape(-1)).float()
y = torch.tensor(np.array((1,1.5,3,2)).reshape(-1)).float()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x, y, likelihood)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

training_iter = 5000

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

logs = {
    'likelihood.noise_covar.raw_noise': [],
    'mean_module.raw_constant': [],
    'covar_module.raw_outputscale': [],
    'covar_module.base_kernel.raw_lengthscale': [],
    'Negative_ExactMarginalLogLikelihood': []
}

for i in range(training_iter):
    model.train()
    likelihood.train()

    optimizer.zero_grad()
    
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")
        logs[name].append(param.data.item())
        
    output = model(x)
        
    loss = -mll(output, y)
    logs['Negative_ExactMarginalLogLikelihood'].append(loss.item())
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    
    optimizer.step()

    if i % 20 == 0:
        model.eval()
        likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 5, 51)
            latent_pred = model(test_x)
            observed_pred = likelihood(latent_pred)

        f, ax = plt.subplots(1, 1, figsize=(8, 6))

        lower, upper = observed_pred.confidence_region()
        ax.plot(x.numpy(), y.numpy(), 'k.', markersize=15)
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), color='k', alpha=0.3)
        ax.set_ylim([-2, 7])
        ax.set_xlim([0, 5])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        save_path = f'data/show_images/{i:05d}.png'
        plt.savefig(save_path)
        plt.close()


f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(logs['Negative_ExactMarginalLogLikelihood'])
plt.title('Negative Exact Marginal Log Likelihood')
plt.savefig('data/show_images/mll.png')
plt.close()

f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(logs['likelihood.noise_covar.raw_noise'])
plt.title('likelihood noise_covar raw_noise')
plt.savefig('data/show_images/likelihood__noise_covar__raw_noise.png')
plt.close()

f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(logs['mean_module.raw_constant'])
plt.title('mean_module raw_constant')
plt.savefig('data/show_images/mean_module__raw_constant.png')
plt.close()

f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(logs['covar_module.raw_outputscale'])
plt.title('covar_module raw_outputscale')
plt.savefig('data/show_images/covar_module__raw_outputscale.png')
plt.close()

f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(logs['covar_module.base_kernel.raw_lengthscale'])
plt.title('covar_module base_kernel raw_lengthscale')
plt.savefig('data/show_images/covar_module__base_kernel__raw_lengthscale.png')
plt.close()