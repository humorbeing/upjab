import gpytorch




class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        # self.mean_module = gpytorch.means.LinearMean(input_size=train_x.size(-1))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.size(-1))
            # gpytorch.kernels.MaternKernel(nu=0.5)
        )
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        # )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)






class Get_GP_model:
    def __init__(self, x_train, y_train):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(x_train, y_train, self.likelihood)
        

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
    

