import torch

class FGM():
    def __init__(self, model: torch.nn.Module, epsilon: int = 1., emb_name: str = 'embeddings'):
        """
        Example:
            >>> FGM(model)
            >>> for batch_input, batch_label in data:
            >>>     # 正常训练
            >>>     loss = model(batch_input, batch_label)
            >>>     loss.backward() # 反向传播，得到正常的grad
            >>>     # 对抗训练
            >>>     fgm.attack() # 在embedding上添加对抗扰动
            >>>     loss_adv = model(batch_input, batch_label)
            >>>     loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            >>>     fgm.restore() # 恢复embedding参数
            >>>     # 梯度下降，更新参数
            >>>     optimizer.step()
            >>>     model.zero_grad()
        """
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}
        assert not set(['attack', 'restore']) & set(dir(self.model))
        self.model.attack = self.attack
        self.model.restore = self.restore

    def attack(self, epsilon=1.):
        epsilon = epsilon if epsilon is not None else self.epsilon
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD():
    def __init__(self, model: torch.nn.Module, epsilon: int = 1., alpha: int = 0.3, emb_name: str = 'embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        """
        Example: 
            >>> PGD(model)
            >>> K = 3
            >>> for batch_input, batch_label in data:
            >>>     # 正常训练
            >>>     loss = model(batch_input, batch_label)
            >>>     loss.backward() # 反向传播，得到正常的grad
            >>>     model.backup_grad()
            >>>     # 对抗训练
            >>>     for t in range(K):
            >>>         model.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
            >>>         if t != K-1:
            >>>             model.zero_grad()
            >>>         else:
            >>>             model.restore_grad()
            >>>         loss_adv = model(batch_input, batch_label)
            >>>         loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            >>>     model.restore() # 恢复embedding参数
            >>>     # 梯度下降，更新参数
            >>>     optimizer.step()
            >>>     model.zero_grad()
        """

        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_name = emb_name
        self.emb_backup = {}
        self.grad_backup = {}
        assert not set(['attack', 'restore', 'backup_grad', 'restore_grad']) & set(dir(self.model))
        self.model.attack = self.attack
        self.model.restore = self.restore
        self.model.backup_grad = self.backup_grad
        self.model.restore_grad = self.restore_grad

    def attack(self, is_first_attack=False, epsilon=None, alpha=None):
        epsilon = epsilon if epsilon is not None else self.epsilon
        alpha = alpha if alpha is not None else self.alpha
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self._project(name, param.data, epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def _project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]