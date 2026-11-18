#!/usr/bin/env python3
"""Train one PAIR router seed on per-chunk features and interface labels.

The target is a soft distribution over the four interfaces built from each
chunk's action error (``tie_aware_target``).  The checkpoint with the lowest
validation routed error is kept.  Train several seeds and pass them together
to ``uniwam.pair.evaluate`` / ``PAIRGate`` for a probability-averaged ensemble.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from groot.vla.model.dreamzero.modules.pair_router import PAIRRouter
from uniwam.pair.data import (
    format_report,
    load_chunk_set,
    routing_report,
    subset,
    tie_aware_target,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", required=True, help="chunk_features.npz from uniwam.pair.features")
    p.add_argument("--labels", required=True, help="predictions.json from uniwam.inference.fourway")
    p.add_argument("--eval-features", help="Separate held-out features (default: the 'test' split)")
    p.add_argument("--eval-labels", help="Labels for --eval-features")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--metric", choices=("mae", "mse"), default="mae")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--tau", type=float, default=0.2, help="Soft-target temperature")
    p.add_argument("--tie-eps", type=float, default=0.01,
                   help="Relative error gap below which a chunk counts as a near tie")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def predict(router: PAIRRouter, inputs: list[torch.Tensor]) -> np.ndarray:
    router.eval()
    with torch.no_grad():
        return router(inputs).argmax(dim=-1).cpu().numpy()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if bool(args.eval_features) != bool(args.eval_labels):
        raise ValueError("--eval-features and --eval-labels must be given together")

    data = load_chunk_set(args.features, args.labels, args.metric)
    train = subset(data, data["split"] == "train")
    val = subset(data, data["split"] == "val")
    if args.eval_features:
        held_out = load_chunk_set(args.eval_features, args.eval_labels, args.metric)
    else:
        held_out = subset(data, data["split"] == "test")
    if not len(train["error"]) or not len(val["error"]):
        raise ValueError("features need non-empty 'train' and 'val' splits")
    print(f"chunks: train {len(train['error'])}  val {len(val['error'])}  "
          f"held-out {len(held_out['error'])}")

    device = torch.device(args.device)
    to_tensors = lambda arrays: [torch.tensor(a, dtype=torch.float32, device=device) for a in arrays]  # noqa: E731
    train_inputs, val_inputs = to_tensors(train["inputs"]), to_tensors(val["inputs"])
    target = torch.tensor(
        tie_aware_target(train["error"], args.tau, args.tie_eps), dtype=torch.float32, device=device
    )
    router = PAIRRouter(
        input_dims=[x.shape[1] for x in train_inputs], hidden=args.hidden, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n = len(target)
    val_rows = np.arange(len(val["error"]))
    best = {"error": float("inf"), "state": None, "epoch": -1}
    for epoch in range(args.epochs):
        router.train()
        perm = torch.randperm(n)
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            logits = router([x[idx] for x in train_inputs])
            loss = -(torch.log_softmax(logits, dim=-1) * target[idx]).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            picks = predict(router, val_inputs)
            val_error = float(val["error"][val_rows, picks].mean())
            if val_error < best["error"]:
                best = {"error": val_error, "state": copy.deepcopy(router.state_dict()), "epoch": epoch + 1}

    router.load_state_dict(best["state"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"pair_router_seed{args.seed}.pt"
    torch.save(router.state_dict(), ckpt)

    summary = {"checkpoint": str(ckpt), "seed": args.seed, "best_epoch": best["epoch"],
               "val_routed": best["error"], "metric": args.metric, "config": vars(args)}
    if len(held_out["error"]):
        report = routing_report(predict(router, to_tensors(held_out["inputs"])), held_out["error"], args.tie_eps)
        summary["held_out"] = report
        print(format_report(report, args.metric))
    (out_dir / f"pair_router_seed{args.seed}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"saved {ckpt} (best epoch {best['epoch']}, val routed {args.metric} {best['error']:.6f})")


if __name__ == "__main__":
    main()
