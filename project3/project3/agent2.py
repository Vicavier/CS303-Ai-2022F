import torch
import torch.nn as nn
import time
from typing import Tuple
from src import FEATURE_DIM, RADIUS, splev, N_CTPS, P, evaluate, N_CLASSES, new_evaluate, compute_traj, generate_game

CLASSIFY_MODEL = './Classify_Model.pth'

class MLP(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n_i, n_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_h, n_o)
    def forward(self, input):
        return self.linear2(self.relu(self.linear1(self.flatten(input))))


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

        
        y_pred = self.myModel.forward(target_features)
        y_predicted_cls = y_pred.argmax(1)
        target_scores = class_scores[y_predicted_cls]
        lr = 0.4
        candidate_ctps = [[torch.rand((N_CTPS-2, 2)) * torch.tensor([N_CTPS-2, 2.]) + torch.tensor([1., -1.]),-999]]
        it = 20
        start = time.time()
        while time.time() - start < 0.15:
            rand_ctps = torch.rand((N_CTPS-2, 2)) * torch.tensor([N_CTPS-2, 2.]) + torch.tensor([1., -1.])
            s = evaluate(compute_traj(rand_ctps), target_pos, target_scores, RADIUS)
            if s > candidate_ctps[0][1]:
                candidate_ctps.insert(0,[rand_ctps,s])
        
        for e in range():
            ctps_inter = candidate_ctps[e][0]
            ctps_inter.requires_grad = True
            for i in range(it):
                if time.time() - start > 0.29:
                    break
                traj = compute_traj(ctps_inter)
                score = new_evaluate(traj, target_pos, target_scores, RADIUS)
                score.backward()
                ctps_inter.data = ctps_inter.data + lr * ctps_inter.grad / torch.norm(ctps_inter.grad)
        
        real_score = evaluate(compute_traj(ctps_inter), target_pos, target_scores, RADIUS)
        
        if real_score > candidate_ctps[0][1]:
            return ctps_inter
        else:
            return candidate_ctps[0][0]
