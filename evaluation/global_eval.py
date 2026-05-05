import numpy as np
from sklearn.metrics import mean_absolute_error


def global_eval(results):
    preds = np.concatenate([r["preds"] for r in results])
    targets = np.concatenate([r["targets"] for r in results])

    mae_model = mean_absolute_error(targets, preds)

    print("\n===== GLOBAL MODEL =====")
    print(f"MAE: {mae_model:.6f}")

    if "baseline" in results[0]:
        baseline = np.concatenate([r["baseline"] for r in results])

        mae_base = mean_absolute_error(targets, baseline)
        skill = 1 - mae_model / mae_base

        corr = np.corrcoef(preds, targets)[0, 1]

        print("\n===== GLOBAL VS BASELINE =====")
        print(f"Baseline MAE: {mae_base:.6f}")
        print(f"Skill: {skill:.4f}")
        print(f"Corr: {corr:.4f}")

        return {
            "mae": mae_model,
            "mae_baseline": mae_base,
            "skill": skill,
            "corr": corr
        }

    return {"mae": mae_model}