"""
ver： AUG 26th 17：00 official release
SAM normalized training, optimizer optimization tool to avoid over-fitting  ICLR 2021 spotlight paper by Google

Introduction：https://mp.weixin.qq.com/s/04VT-ldd0-XEkhEW6Txl_A
Third party implementation from：https://pub.towardsai.net/we-dont-need-to-worry-about-overfitting-anymore-9fb31a154c81

Paper：Sharpness-aware Minimization for Efficiently Improving Generalization
Link：https://arxiv.org/abs/2010.01412

Calculation principle:
During training, when the optimizer updates the model parameter w, it can be divided into four steps as a whole:

 (1) Calculate gradient G on batch data S based on parameter w.
 
 (2) Solve the dual norm of G, update parameters according to the direction of dual vector, and obtain the auxiliary model under the w+ε system.

 (3) Based on the auxiliary model under the parameter w+ε, calculate gradient G' for S
 
 (4) Update original model parameter w with G'

Usage examples:
from sam import SAM
...
model = YourModel()
base_optimizer = torch.optim.SGD  # Pass in an optimizer template
optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)  # Optimizer parameters
...
for input, output in data:

  # first forward-backward pass, Calculate first round loss, which is the same as normal
  # The loss in the first round is run out of the real model, which is what we need in statistics. 
  # The loss in the second round is not the loss of the real model (it is fromn auxiliary model), so it does not need to be used in traditional statistical loss.
  output = model(input)
  loss = loss_function(output, labels)  # use this loss for any training statistics！！！！
  loss.backward()  # Model back propagation, record original gradient.

  # SAM class from step1 calculated the 'SAM gradient'
  optimizer.first_step(zero_grad=True)  # The first round of opt used "SAM gradient" to update the original model parameter system, and now the model has become an auxiliary model.
  # Step1 records and saves the method of returning to the original model parameter system

  # second forward-backward pass  In the second round, the auxiliary model (the updated model in step1) is propagated forward and backward first
  output2 = model(input)  # Use output2 to ensure that the calculation graph is an auxiliary model (that is, the updated model of step1), otherwise there will be a bunch of bugs.

  # Due to the newly added calculation graph, the calculation time and the memory usage increased?

  loss_function(output2, labels).backward()  # make sure to do a full forward pass. Auxiliary model backpropagation, record the updated gradient
  optimizer.second_step(zero_grad=True)  # In the second round, replace the original model parameters first, and then base opt updates the original model parameter system with the update direction of the auxiliary model.
...


"""
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):  # step1 Generate auxiliary model, modify the original model parameters to turn it into an auxiliary model, and record how it changed so as to restore
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)  # Gradient influence nearby

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)  # After considering the influence of nearby gradients, determine the "SAM gradient" required for the parameter changes of the auxiliary model
                p.add_(e_w)  # climb to the local maximum "w + e(w)". Inplace parameters updated! Thus original model changed into auxiliary model
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):  # step2 First modify the auxiliary model parameters to change it back to the original model,
        # After that, the gradient of the original model based on the auxiliary model is updated with base_optimizer.

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                # Auxiliary model parameter restoration, return to the original 'model get back to "w" from "w + e(w)"', note that this is also inplaced! !

        self.base_optimizer.step()  # 用base_optimizer对原模型进行参数更新 do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None]), p=2)
        return norm
