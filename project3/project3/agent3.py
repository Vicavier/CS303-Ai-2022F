import torch
import torch.nn as nn
import time
from typing import Tuple
from functorch import vmap


# ==============全局变量==================
CLASSIFY_MODEL = 'project3/Classify_Model_2.pth'
FEATURE_DIM = 256
RADIUS = 0.3
N_CTPS = 5
P = 3
N_CLASSES = 10

# ==============定义模型==================
class MLP(nn.Module):
    def __init__(self, n_i, n_h_1, n_h_2, n_o):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n_i, n_h_1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_h_1, n_h_2)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(n_h_2,n_o)
    def forward(self, input):
        return self.linear3(self.relu(self.linear2(self.relu(self.linear1(self.flatten(input))))))

class Agent:

    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """
        global FEATURE_DIM, N_CLASSES, CLASSIFY_MODEL
        # prepare your agent here
        self.myModel = MLP(FEATURE_DIM, FEATURE_DIM // 2,FEATURE_DIM // 4, N_CLASSES)
        self.myModel.load_state_dict(torch.load(CLASSIFY_MODEL))
        self.myModel.eval()
        torch.manual_seed(20)
        torch.cuda.manual_seed_all(20)

    def vmap_fun(self, ctps_inter):
        return evaluate(compute_traj(ctps_inter), self.target_pos, self.target_scores, RADIUS)

    def get_action(self,
        target_pos: torch.Tensor,
        target_features: torch.Tensor,
        class_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the parameters required to fire a projectile. 
        
        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets. 
            target_features: features of shape `(N, d)`.
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
        Return: Tensor of shape `(N_CTPS-2, 2)`
            the second to the second last control points
        """
        assert len(target_pos) == len(target_features)
        
        # compute the firing speed and angle that would give the best score.
        start = time.time()
        y_pred = self.myModel.forward(target_features)
        y_predicted_cls = y_pred.argmax(1)

        self.target_scores = class_scores[y_predicted_cls]
        self.target_pos = target_pos
        # BATCH = 10
        it = 20
        lr = 0.4
        ctps = [None,-2e9]
        while True:
            
            ctps_inter =  torch.randn((200, N_CTPS-2, 2)) * torch.tensor([N_CTPS-2, 2.]) + torch.tensor([1., -1.])
            # ctps_inter = torch.Tensor([[random.uniform(2,6),random.uniform(0,3)],[random.uniform(-1,0),random.uniform(-3,0)],[random.uniform(2,6),random.uniform(0,3)]])
            # ctps_inter = torch.Tensor([[1.25,0.5],[3.0,-0.5],[4.0,-0.5]])
            
            score = vmap(self.vmap_fun)(ctps_inter)
            print(score)
            
            
        
        # print(f'score: {ctps[epoch - 1][0]}')
        # print(class_scores)
        # print(f'number of positive position: {positive}')
        # print(f'time: {time.time() - start}')
        return ctps[0]

def compute_traj(ctps_inter: torch.Tensor):
    """Compute the discretized trajectory given the second to the second control points"""
    t = torch.linspace(0, N_CTPS-P, 100, device=ctps_inter.device)
    knots = torch.cat([
        torch.zeros(P, device=ctps_inter.device), 
        torch.arange(N_CTPS+1-P, device=ctps_inter.device), 
        torch.full((P,), N_CTPS-P, device=ctps_inter.device),
    ])
    ctps = torch.cat([
        torch.tensor([[0., 0.]], device=ctps_inter.device),     #ctps:[[0,0],...,[5,0]]
        ctps_inter,
        torch.tensor([[N_CTPS, 0.]], device=ctps_inter.device)
    ])
    return splev(t, knots, ctps, P)


def evaluate(
    traj: torch.Tensor, 
    target_pos: torch.Tensor, 
    target_scores: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    cdist = torch.cdist(target_pos, traj) # see https://pytorch.org/docs/stable/generated/torch.cdist.html
    d = cdist.min(-1).values
    hit = (d < radius)
    value = torch.sum(hit * target_scores, dim=-1)
    return value


def sigmoid(d:torch.Tensor):
    return 1. / (1 + torch.exp(40* (d-0.3)))


def new_evaluate(
    traj: torch.Tensor, 
    target_pos: torch.Tensor, 
    target_scores: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    cdist = torch.cdist(target_pos, traj)
    d = cdist.min(-1).values
    hit = sigmoid(d)
    # hit = d <= radius
    # d[hit] = 1
    # d[~hit] = radius / d[~hit]
    value = torch.sum(hit * target_scores, dim=-1)
    return value

def splev(
    x: torch.Tensor, 
    knots: torch.Tensor, 
    ctps: torch.Tensor, 
    degree: int, 
    der: int=0
) -> torch.Tensor:
    if der == 0:
        return _splev_torch_impl(x, knots, ctps, degree)
    else:
        assert der <= degree, "The order of derivative to compute must be less than or equal to k."
        n = ctps.size(-2)
        ctps = (ctps[..., 1:, :]-ctps[..., :-1, :])/(knots[degree+1:degree+n]-knots[1:n]).unsqueeze(-1)
        return degree * splev(x, knots[..., 1:-1], ctps, degree-1, der-1)


def _splev_torch_impl(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int):
    """
        x: (t,)
        t: (m, ) 
        c: (n_ctps, dim)
    """
    # assert t.size(0) == c.size(0) + k + 1, f"{len(t)} != {len(c)} + {k} + {1}" # m= n + k + 1

    x = torch.atleast_1d(x)
    assert x.dim() == 1 and t.dim() == 1 and c.dim() == 2, f"{x.shape}, {t.shape}, {c.shape}"

    n = c.size(0)
    u = (torch.searchsorted(t, x)-1).clip(k, n-1).unsqueeze(-1)
    x = x.unsqueeze(-1)
    d = c[u-k+torch.arange(k+1, device=c.device)].contiguous()
    for r in range(1, k+1):
        j = torch.arange(r-1, k, device=c.device) + 1
        t0 = t[j+u-k]
        t1 = t[j+u+1-r]
        alpha = ((x - t0) / (t1 - t0)).unsqueeze(-1)
        d[:, j] = (1-alpha)*d[:, j-1] + alpha*d[:, j]
    return d[:, k]
# P = 3
# N_CTPS = 5
# N_TARGETS = 40
# N_CLASSES = 10
# RADIUS = 0.3


# data0 = torch.load('./data.pth')
# X = data0['feature']
# y = data0['label']
# print(f'X.shape = {X.shape}')
# print(f'X.shape = {y.shape}')
# X_train = X[:48000]
# y_train = y[:48000]
# X_test = X[48000:]
# y_test = y[48000:]
# a = Agent()
# feature = X_test[:40,:]
# class_scores = torch.randint(-N_CLASSES, N_CLASSES, (N_CLASSES,))
# target_pos = torch.rand((N_TARGETS, 2)) * torch.tensor([N_CTPS-2, 2.]) + torch.tensor([1., -1.])

# ctps = a.get_action(target_pos,feature,class_scores)
# print(ctps)