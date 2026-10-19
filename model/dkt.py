"""Minimal Deep Knowledge Tracing model (Piech et al., 2015).

Convention: logits[:, t, :] is the predicted probability vector for the
*next* interaction (position t+1), given the history 0..t.
"""

import numpy as np
import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(self, num_skills, hidden_size=64, dropout=0.2):
        super().__init__()
        self.num_skills = num_skills
        self.lstm = nn.LSTM(
            input_size=num_skills + 1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, num_skills)

    def forward(self, skill_ids, correct, valid):
        """skill_ids (B, L) long, correct (B, L) float, valid (B, L) bool.

        Returns logits (B, L, num_skills): raw (pre-sigmoid) probabilities of a
        correct next answer, one column per skill.
        """
        batch, length = skill_ids.shape
        one_hot = torch.zeros(batch, length, self.num_skills, device=skill_ids.device)
        one_hot.scatter_(2, skill_ids.unsqueeze(-1), 1.0)
        one_hot = one_hot * valid.unsqueeze(-1)  # zero out padded positions
        x = torch.cat([one_hot, correct.unsqueeze(-1)], dim=-1)  # (B, L, num_skills+1)
        outputs, _ = self.lstm(x)               # (B, L, hidden_size) per step
        logits = self.out(self.dropout(outputs))  # (B, L, num_skills)
        return logits


def masked_loss(logits, skill_ids, correct, valid):
    """BCE on the pairs (step t predicts skill of step t+1, its correctness).

    Padded and unobserved positions are masked, so the loss only counts
    interactions that really happened.
    """
    pred_pos = valid[:, :-1]   # history exists up to t
    target_pos = valid[:, 1:]  # a real interaction exists at t+1
    pair_mask = (pred_pos & target_pos).float()          # (B, L-1)
    step_logits = logits[:, :-1]                          # (B, L-1, num_skills)
    next_skill = skill_ids[:, 1:]                         # (B, L-1)
    next_correct = correct[:, 1:]                         # (B, L-1)
    selected = torch.gather(step_logits, 2, next_skill.unsqueeze(-1)).squeeze(-1)
    bce = nn.functional.binary_cross_entropy_with_logits(
        selected, next_correct, reduction="none"
    )
    return (bce * pair_mask).sum() / pair_mask.sum()


@torch.no_grad()
def collect_predictions(model, loader, device="cpu"):
    """Run the model over every sequence and return flat evaluation pairs.

    Returns numpy arrays (probs, labels, skills, skill_seen_before), where
    skill_seen_before[k] is True if the tested skill already appeared earlier
    in the same student's history. The paper excludes first observations,
    because the model cannot know an unseen skill.
    """
    model.eval()
    probs_list, labels_list, skills_list, seen_list = [], [], [], []
    for batch in loader:
        skill_ids = batch["skill_ids"].to(device)
        correct = batch["correct"].to(device)
        valid = batch["valid"].to(device)
        logits = model(skill_ids, correct, valid)
        pair_mask = valid[:, :-1] & valid[:, 1:]
        selected = torch.gather(
            logits[:, :-1], 2, skill_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        probs = torch.sigmoid(selected)
        b_idx, t_idx = pair_mask.nonzero(as_tuple=True)
        skills_tested = skill_ids[b_idx, t_idx + 1]
        for b, t, s in zip(b_idx.tolist(), t_idx.tolist(), skills_tested.tolist()):
            probs_list.append(probs[b, t].item())
            labels_list.append(float(correct[b, t + 1].item()))
            skills_list.append(s)
            seen_list.append(s in skill_ids[b, : t + 1].tolist())
    return (
        np.array(probs_list),
        np.array(labels_list),
        np.array(skills_list),
        np.array(seen_list, dtype=bool),
    )
