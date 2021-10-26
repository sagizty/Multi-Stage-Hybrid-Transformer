"""
版本： 8月26日 17：00
SAM范化性训练，避免过拟合的优化器优化工具  ICLR 2021 spotlight paper by Google

介绍：https://mp.weixin.qq.com/s/04VT-ldd0-XEkhEW6Txl_A
第三方实现来自：https://pub.towardsai.net/we-dont-need-to-worry-about-overfitting-anymore-9fb31a154c81

论文：Sharpness-aware Minimization for Efficiently Improving Generalization
链接：https://arxiv.org/abs/2010.01412

计算原理：
在训练过程中，优化器更新模型参数w时，整体上可以分为四个步骤：

（1）基于参数 w 对 batch data S 计算 gradient G 。

（2）求解 G 的 dual norm，依照 dual vector 方向更新参数，得到 w+ε体系下的辅助模型。

（3）基于参数 w+ε 下的辅助模型，对 S 计算 gradient G’ 。

（4）用 G’ 更新原本的模型的参数 w 。


使用例子：
from sam import SAM
...
model = YourModel()
base_optimizer = torch.optim.SGD  # 传入一个优化器模板
optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)  # 优化器参数
...
for input, output in data:

  # first forward-backward pass，计算第一轮loss，这个和普通的一样
  # 第一轮的loss是真实模型跑出来的，我们统计中需要的loss就是它，第二轮loss不是真实的模型的loss（是辅助模型的），所以不需要用在传统统计loss中
  output = model(input)
  loss = loss_function(output, labels)  # use this loss for any training statistics！！！！
  loss.backward()  # 模型反向传播，记录原梯度。

  # step1 的SAM类计算了“SAM梯度”
  optimizer.first_step(zero_grad=True)  # 第一轮opt用“SAM梯度”对原模型参数体系进行了更新，现在模型变成了辅助模型，
  # step1记录保存了回到原模型参数体系的变化方法

  # second forward-backward pass  第二轮先对辅助模型（step1更新后的模型）正向、反向传播
  output2 = model(input)  # 用output2 确保计算图是辅助模型（即step1更新后的模型），不然有一堆bug。

  # 由于新增了计算图，因此计算时间增加显存占用也增加？

  loss_function(output2, labels).backward()  # make sure to do a full forward pass 辅助模型反向传播，记录更新梯度
  optimizer.second_step(zero_grad=True)  # 第二轮，先原模型参数替换回去，之后base opt以辅助模型的更新方向对原模型参数体系进行更新
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
    def first_step(self, zero_grad=False):  # step1 生成辅助模型，对原模型参数进行修改把他变成辅助模型，同时记录怎么变的，以便还原
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)  # 附近的梯度影响

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)  # 考虑附近的梯度影响之后，确定辅助模型的参数变化需要的“SAM梯度”
                p.add_(e_w)  # climb to the local maximum "w + e(w)"  inplace参数更新！ 因此是 原模型 变成了 辅助模型
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):  # step2 先对辅助模型参数进行修改把他变回原模型，
        # 之后对原模型基于辅助模型的梯度用base_optimizer进行参数更新

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                # 辅助模型参数还原，回到原模型 get back to "w" from "w + e(w)"，注意这个也是inplace的！！

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
