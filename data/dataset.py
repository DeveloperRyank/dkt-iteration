"""Sequence data format and batching.

A sequence is a dict: {"skills": [int], "correct": [int]}
skills[k] in [0, num_skills) is the skill of interaction k; correct[k] is 0/1.
Both lists have the same length.
"""

import torch
from torch.utils.data import Dataset


def collate_fn(seqs):
    """Pad a list of sequences to a batch of tensors.

    Returns dict with:
      skill_ids: (B, L) long   -- padded with 0
      correct:   (B, L) float  -- padded with 0
      valid:     (B, L) bool   -- True for real interactions only
    """
    lengths = [len(s["skills"]) for s in seqs]
    max_len = max(lengths)
    batch = len(seqs)
    skill_ids = torch.zeros(batch, max_len, dtype=torch.long)
    correct = torch.zeros(batch, max_len)
    valid = torch.zeros(batch, max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = lengths[i]
        skill_ids[i, :n] = torch.tensor(s["skills"], dtype=torch.long)
        correct[i, :n] = torch.tensor(s["correct"], dtype=torch.float)
        valid[i, :n] = True
    return {"skill_ids": skill_ids, "correct": correct, "valid": valid}


class KTSequenceDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


def training_pairs(seqs):
    """Flat (skills, correct) lists of all interactions, in order."""
    skills, correct = [], []
    for s in seqs:
        skills.extend(s["skills"])
        correct.extend(s["correct"])
    return torch.tensor(skills), torch.tensor(correct, dtype=torch.float)


def make_loader(seqs, batch_size, shuffle=False):
    return torch.utils.data.DataLoader(
        KTSequenceDataset(seqs),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
