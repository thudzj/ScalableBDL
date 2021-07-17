import torch

class PriorRegularizor:
    def __init__(self, model, decay, num_data, num_mc_samples, 
                 posterior_type='mf', MOPED=False):
        super(PriorRegularizor, self).__init__()

        self.model = model
        self.decay = decay
        self.num_data = num_data
        self.num_mc_samples = num_mc_samples
        self.posterior_type = posterior_type
        self.MOPED = MOPED

        if MOPED:
            self.init_mus = {}
            for name, param in model.named_parameters():
                if '_mu' in name:
                    self.init_mus[name] = param.data.clone().detach()

    @torch.no_grad()
    def step(self,):
        for name, param in self.model.named_parameters():
            if self.posterior_type == "mf" or self.posterior_type == "mean_field":
                if '_psi' in name:
                    param.grad.data.add_((param*2).exp(), alpha=self.decay).sub_(1./self.num_data)
                else:
                    if self.MOPED:
                        param.grad.data.add_(param - self.init_mus[name], alpha=self.decay)
                    else:
                        param.grad.data.add_(param, alpha=self.decay)
            elif self.posterior_type is None:
                param.grad.data.add_(param, alpha=self.decay)
            else:
                raise NotImplementedError
