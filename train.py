"""Train the DKT and compare against baselines.

Usage:
  python train.py --data synthetic --epochs 20
  python train.py --data assistments --epochs 5
"""

import argparse
import time

import numpy as np
import torch

from baseline import fit_per_skill_rate, predict_per_skill_rate
from config import Config
from data.dataset import make_loader, training_pairs
from data.synthetic import generate_sequences, split_by_student
from evaluate import format_report, report
from model.dkt import DKT, collect_predictions, masked_loss


def get_data(name, cfg):
    if name == "synthetic":
        data = generate_sequences(seed=cfg.seed)
    else:
        from data.assistments import load_assistments
        data, _ = load_assistments(verbose=(name == "assistments"))
    return data


def train_model(model, train_loader, cfg):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for epoch in range(cfg.epochs):
        total, batches = 0.0, 0
        for batch in train_loader:
            logits = model(batch["skill_ids"], batch["correct"], batch["valid"])
            loss = masked_loss(logits, batch["skill_ids"], batch["correct"], batch["valid"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        avg = total / batches
        history.append(avg)
        print(f"epoch {epoch + 1:2d}/{cfg.epochs}  train loss {avg:.4f}", flush=True)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "assistments"],
                        default="synthetic")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--save", type=str, default=None,
                        help="Save model checkpoint to this path")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.hidden is not None:
        cfg.hidden_size = args.hidden
    if args.lr is not None:
        cfg.lr = args.lr
    if args.dropout is not None:
        cfg.dropout = args.dropout
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data = get_data(args.data, cfg)
    train_seqs, test_seqs = split_by_student(data["sequences"], seed=cfg.seed)
    print(f"{args.data}: {len(train_seqs)} train / {len(test_seqs)} test students, "
          f"{data['num_skills']} skills")

    model = DKT(data["num_skills"], cfg.hidden_size, cfg.dropout)
    train_loader = make_loader(train_seqs, cfg.batch_size, shuffle=True)
    test_loader = make_loader(test_seqs, cfg.batch_size, shuffle=False)
    t0 = time.perf_counter()
    history = train_model(model, train_loader, cfg)
    train_seconds = round(time.perf_counter() - t0, 1)

    probs, labels, skills, seen = collect_predictions(model, test_loader)
    keep = seen  # protocol: exclude each skill's first observation
    rows = [report("DKT (first-obs excluded)", probs[keep], labels[keep]),
            report("DKT (all pairs)", probs, labels)]

    tr_skills, tr_labels = training_pairs(train_seqs)
    rates, global_rate = fit_per_skill_rate(tr_skills.numpy(), tr_labels.numpy(),
                                            data["num_skills"])
    base_probs = predict_per_skill_rate(rates, global_rate, skills[keep])
    rows.append(report("per-skill rate baseline", base_probs, labels[keep]))
    rows.append(report("random guess", np.random.default_rng(0).random(int(keep.sum())),
                       labels[keep]))

    print("\ntest results:")
    for r in rows:
        print(format_report(r))

    print(f"\nloss decreased: {history[0]:.4f} -> {history[-1]:.4f}")

    from plots import save_plots
    save_plots(history, [("DKT", probs[keep], labels[keep]),
                         ("per-skill rate baseline", base_probs, labels[keep])],
               prefix=args.data)

    if args.save:
        torch.save({
            "state_dict": model.state_dict(),
            "num_skills": data["num_skills"],
            "hidden_size": cfg.hidden_size,
            "dropout": cfg.dropout,
            "epochs": cfg.epochs,
            "data": args.data,
            "test_auc": rows[0]["auc"],
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
            "train_seconds": train_seconds,
        }, args.save)
        print(f"checkpoint saved to {args.save}")


if __name__ == "__main__":
    main()
