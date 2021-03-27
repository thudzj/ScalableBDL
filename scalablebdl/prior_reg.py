import torch



class PriorRegularizor:
    def __init__(self, model, decay, num_data, num_mc_samples, posterior_type, MOPED):
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
    def get_param_by_name(self, name):
        o = self.model
        for i in name.split('.'):
            o = getattr(o, i)
        return o

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
            elif self.posterior_type == "emp" or self.posterior_type == "empirical":
                if 'weights' in name or 'biases' in name:
                    param.grad.data.add_(param, alpha=self.decay/self.num_mc_samples)
                else:
                    param.grad.data.add_(param, alpha=self.decay)
            elif self.posterior_type == "lr" or self.posterior_type == "low_rank":
                if 'weight_mu' in name:
                    b = self.get_param_by_name(name.replace("weight_mu", "in_perturbations"))
                    a = self.get_param_by_name(name.replace("weight_mu", "out_perturbations"))
                    m_ = torch.bmm(a, b)
                    param.grad.data.add_((m_**2).mean(0).view_as(param) * param, alpha=self.decay)
                    m_ = m_.mul_((param**2).flatten(1, -1))
                    a.grad.data.add_(torch.bmm(m_, b.permute(0, 2, 1)), alpha=self.decay/self.num_mc_samples)
                    b.grad.data.add_(torch.bmm(a.permute(0, 2, 1), m_), alpha=self.decay/self.num_mc_samples)
                elif not 'perturbations' in name:
                    param.grad.data.add_(param, alpha=self.decay)
            elif self.posterior_type is None:
                param.grad.data.add_(param, alpha=self.decay)
            else:
                raise NotImplementedError
