import torch
import torchmetrics
import numpy as np
from sklearn.metrics import mean_absolute_error


def evaluate_model(model, test_loader, device, baseline_series=None):

    mse_metric = torchmetrics.MeanSquaredError().to(device)

    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)

            outputs = model(seqs).squeeze()

            mse_metric.update(outputs, labels)

            preds.append(outputs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    mse = mse_metric.compute().item()
    mae = mean_absolute_error(targets, preds)

    result = {
        "mse": mse,
        "mae": mae,
        "preds": preds,
        "targets": targets,
    }

    print("\n===== MODEL METRICS =====")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    if baseline_series is not None:
        baseline = np.asarray(baseline_series)

        # alignment
        min_len = min(len(targets), len(baseline))
        targets = targets[-min_len:]
        preds = preds[-min_len:]
        baseline = baseline[-min_len:]

        baseline = np.nan_to_num(baseline, nan=np.nanmean(baseline))

        mae_base = mean_absolute_error(targets, baseline)
        skill = 1 - mae / mae_base
        corr = np.corrcoef(preds, targets)[0, 1]

        print("\n===== VS BASELINE =====")
        print(f"Baseline MAE: {mae_base:.6f}")
        print(f"Skill: {skill:.4f}")
        print(f"Corr: {corr:.4f}")

        result.update({
            "mae_baseline": mae_base,
            "skill": skill,
            "corr": corr,
            "baseline": baseline
        })

    return result