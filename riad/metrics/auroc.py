import matplotlib.pyplot as plt
from numpy import ndarray as NDArray
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auroc(epoch: int, ep_reconst: NDArray, ep_gt: NDArray) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)

    Args:
        epoch (int): Current epoch
        ep_reconst (NDArray): Reconstructed images in a current epoch
        ep_gt (NDArray): Ground truth masks in a current epoch

    Returns:
        float: AUROC score
    """

    num_data = len(ep_reconst)
    y_score = ep_reconst.reshape(num_data, -1).max(axis=1)  # y_score.shape -> (num_data,)
    y_true = ep_gt.reshape(num_data, -1).max(axis=1)  # y_true.shape -> (num_data,)

    score = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    plt.plot(fpr, tpr, marker="o", color="k", label=f"AUROC Score: {round(score, 3)}")
    plt.xlabel("FPR: FP / (TN + FP)", fontsize=14)
    plt.ylabel("TPR: TP / (TP + FN)", fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"epochs/{epoch}/roc_curve.png")
    plt.close()

    return score
