import torch
import torch.distributed as dist

class LossManager:
    def __init__(self,eval=False):
        self.loss_terms = {}
        self.total_loss = 0.0
        self.eval=eval
        self.reset_accumulate()

    def add_loss_terms(self, dic):
        self.loss_terms |= dic

    def reset_accumulate(self):
        self.count = 0
        self.loss_dict = {
            k:0 for k in self.loss_terms.keys()
        }
    
    def update(self, dic: dict):
        for k, v in dic.items():
            if k in self.loss_terms.keys():
                v=v * self.loss_terms[k]
                if v.isnan().item():
                    raise ValueError(f"Loss {k} is NaN!")
                if not self.eval:
                    # for training
                    self.total_loss += v
                # for logging
                self.loss_dict[k] += v.detach().cpu().item()

    def step(self):
        self.count += 1
        if not self.eval:
            self.total_loss.backward()
        self.total_loss = 0.0
        
    def reduce_loss_dict(self):
        """
        用 all_reduce 把多 GPU 上的 loss 同步为平均值, 然后关于 self.count 归一化。
        """
        with torch.no_grad():
            keys = list(self.loss_dict.keys())
            values = torch.tensor([self.loss_dict[k] for k in keys], device=torch.cuda.current_device())
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= dist.get_world_size()
            self.loss_dict = dict(zip(keys, values.cpu().tolist()))
            self.loss_dict = {k: v / self.count for k, v in self.loss_dict.items()}
        
    def get_log(self):
        assert self.count > 0, "Call .step() at least once before .get_log()"
        if not self.eval:
            return {"train_"+k:v for k,v in self.loss_dict.items()}|{"train_total_loss":sum(self.loss_dict.values())}
        else:
            return {"eval_"+k:v for k,v in self.loss_dict.items()}|{"eval_total_loss":sum(self.loss_dict.values())}