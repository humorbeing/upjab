import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy



class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class Get_GP_model:
    def __init__(self, num_inducing_point, x_train):
        inducing_points = x_train[:num_inducing_point, :]
        self.model = GPModel(inducing_points=inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def to(self, device):
        self.model.to(device)
        self.likelihood.to(device)
        return self
    
    def train(self):
        self.model.train()
        self.likelihood.train()
        return self
    
    def eval(self):
        self.model.eval()
        self.likelihood.eval()
        return self
    

