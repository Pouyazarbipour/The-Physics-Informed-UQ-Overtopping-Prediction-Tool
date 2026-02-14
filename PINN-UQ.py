"""
Staged physics-informed surrogate with aleatoric + epistemic uncertainty (MC Dropout)
for wave overtopping prediction using CLASH data.

Causal Discovery and Integration (Methods-ready summary)
---------------------------------------------------------
We estimate a directed acyclic graph (DAG) over hydraulic predictors and overtopping
response q with a configurable backend: DAGMA (preferred, linear/nonlinear) or
NOTEARS-linear fallback. DAGMA uses a path-following strategy for smooth DAG learning,
while NOTEARS imposes exact smooth acyclicity h(W)=tr(expm(W∘W))−d=0.
This dual-backend design preserves reproducibility in environments where DAGMA is not
installed, while prioritizing DAGMA when available for improved optimization behavior.

To avoid data leakage, the DAG is learned only on the training split. The inferred
parents of q are then used in causal-aware training via:
1) Causal feature masking (optional): non-parents are zeroed at model input.
2) Causal gradient regularization (optional): penalize \|∂q/∂x_i\|^2 for i ∉ Pa(q),
   discouraging reliance on non-causal shortcuts and improving out-of-distribution
   robustness and uncertainty behavior.
"""

import os
import math
import time
import random
import json
from typing import Dict, List, Tuple

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import probplot

try:
    from dagma.linear import DagmaLinear
    from dagma.nonlinear import DagmaMLP, DagmaNonlinear
    DAGMA_AVAILABLE = True
except Exception:
    DAGMA_AVAILABLE = False

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'legend.framealpha': 0.92,
    'savefig.dpi': 400,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.85,
    'grid.alpha': 0.15,
    'grid.linestyle': '--',
    'grid.linewidth': 0.4,
})

ROOT_OUT = os.path.join(os.getcwd(), "surr_outputs_overtopping_phys_epistemic")
PLOTS_DIR = os.path.join(ROOT_OUT, "plots")
DATA_DIR  = os.path.join(ROOT_OUT, "plot_data")
REPORTS_DIR = os.path.join(ROOT_OUT, "reports")
CAUSAL_DIR = os.path.join(ROOT_OUT, "causal_reports")
CKPT_DIR = os.path.join(ROOT_OUT, "checkpoints")

def ensure_dir(path: str, name: str = ""):
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"   Created directory: {name} → {path}")
        else:
            print(f"   Directory exists: {name} → {path}")
    except Exception as e:
        print(f"!!! FAILED to create {name}: {path}")
        print(f"Error: {e}")
        raise RuntimeError(f"Cannot create directory: {path}. Check permissions or path.")

print("\n=== Checking / Creating output directories ===")
ensure_dir(ROOT_OUT, "Root output")
ensure_dir(PLOTS_DIR, "Plots")
ensure_dir(DATA_DIR, "Plot data (excel)")
ensure_dir(REPORTS_DIR, "Reports & metrics")
ensure_dir(CAUSAL_DIR, "Causal analysis")
ensure_dir(CKPT_DIR, "Checkpoints (models)")
print("All directories are ready.\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CFG = {
    "lr": 0.00013607280572255773,
    "batch_size": 16,
    "epochs": 400,
    "mlp_hidden": 256,
    "mlp_layers": 3,
    "weight_decay": 1.386905837978936e-05,
    "patience": 30,
    "print_every": 1,
    "w_q": 1.0,
    "w_phys_base": 1e-5,
    "warm_phys_start": 10,
    "warm_phys_end": 40,
    "mc_samples": 150,
    "dropout_p": 0.20,
    "use_causal_mask": False,
    "use_causal_grad_penalty": False,
    "w_causal": 0,
    "causal_edge_threshold": 0.02,
    "notears_lambda1": 1e-2,
    "notears_max_iter": 25,
    "notears_inner_steps": 250,
    "notears_lr": 1e-2,
    "notears_h_tol": 1e-8,
    "notears_rho_max": 1e16,
}

G = 9.81
EPS = 1e-8

FEATURE_COLUMNS = [
    'beta', 'hm0', 'h', 'tm-10_toe', 'bt', 'ht', 'gama_f',
    'cotad', 'cotau', 'hb', 'rc', 'b', 'tanb', 'ac', 'gc'
]
TARGET_NAME = "q"

MIN_Q_PLOT = 1e-7
MAX_Q_PLOT = 0.2

COLOR_OBSERVED   = '#2ca02c'
COLOR_PREDICTED  = '#1f77b4'
COLOR_TEST       = '#d62728'
COLOR_EUROTOP    = '#ff7f0e'
COLOR_EPI        = '#9467bd'
COLOR_ALE        = '#8c564b'
COLOR_CI         = '#ffdddd'

def deg2rad(d):
    return d * math.pi / 180.0

class Scaler:
    def __init__(self, log=False):
        self.log = log
        self.mean = 0.0
        self.std = 1.0

    def fit(self, arr):
        a = np.asarray(arr).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            self.mean, self.std = 0.0, 1.0
            return
        if self.log:
            a = np.log(np.where(a <= 0, 1e-6, a) + 1e-8)
        self.mean = float(np.mean(a))
        self.std = float(np.std(a)) + 1e-9

    def transform(self, arr):
        a = np.asarray(arr)
        if self.log:
            a = np.log(np.where(a <= 0, 1e-6, a) + 1e-8)
        return (a - self.mean) / self.std

    def inverse(self, arr):
        a = np.asarray(arr)
        val = a * self.std + self.mean
        if self.log:
            return np.exp(val)
        return val

    def inverse_torch(self, x, device=None):
        if device is None:
            device = x.device
        mean_t = torch.tensor(self.mean, dtype=x.dtype, device=device)
        std_t = torch.tensor(self.std, dtype=x.dtype, device=device)
        val = x * std_t + mean_t
        if self.log:
            return torch.exp(val)
        return val

def parse_clash(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    required = [TARGET_NAME] + FEATURE_COLUMNS
    for r in required:
        if r not in df.columns:
            raise RuntimeError(f"CLASH CSV missing column: {r}")
    events = df[required].copy()
    q_values = events[TARGET_NAME].values.astype(np.float32)
    events = events.drop(columns=[TARGET_NAME])
    return events.reset_index(drop=True), q_values

class OvertoppingSurrogateDataset(Dataset):
    def __init__(self, events_df, q_arr, scalers):
        self.events = events_df.reset_index(drop=True)
        self.q = q_arr.astype(np.float32)
        self.scalers = scalers

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        row = self.events.iloc[idx]
        scalars_raw = np.array([float(row[c]) for c in FEATURE_COLUMNS], dtype=np.float32)
        scalars = (scalars_raw - self.scalers['scalars']['mean']) / (self.scalers['scalars']['std'] + 1e-9)
        q_t = self.scalers['q'].transform(self.q[idx])
        return {
            "scalars": torch.from_numpy(scalars),
            "scalars_raw": torch.from_numpy(scalars_raw),
            "target": torch.tensor(q_t)
        }

# ---------- Causal discovery backends ----------
def _acyclicity_torch(W: torch.Tensor) -> torch.Tensor:
    d = W.shape[0]
    M = W * W
    return torch.trace(torch.matrix_exp(M)) - d


def notears_linear_torch(
    X: np.ndarray,
    lambda1: float = 1e-2,
    max_iter: int = 20,
    inner_steps: int = 200,
    lr: float = 1e-2,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    seed: int = 42,
) -> np.ndarray:
    """Linear NOTEARS with augmented Lagrangian and exact acyclicity constraint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_t = torch.tensor(X, dtype=torch.float64)
    n, d = X_t.shape
    W = torch.zeros((d, d), dtype=torch.float64, requires_grad=True)

    rho = 1.0
    alpha = 0.0
    h_prev = np.inf
    eye_mask = (1.0 - torch.eye(d, dtype=torch.float64))

    for _ in range(max_iter):
        optimizer = torch.optim.Adam([W], lr=lr)
        for _ in range(inner_steps):
            optimizer.zero_grad()
            W_eff = W * eye_mask
            fit = 0.5 / n * torch.sum((X_t - X_t @ W_eff) ** 2)
            l1 = lambda1 * torch.sum(torch.abs(W_eff))
            h = _acyclicity_torch(W_eff)
            loss = fit + l1 + 0.5 * rho * (h ** 2) + alpha * h
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            W.data *= eye_mask
            h_new = float(_acyclicity_torch(W * eye_mask).item())

        if h_new <= h_tol or rho >= rho_max:
            break
        if h_new > 0.25 * h_prev:
            rho *= 10.0
        else:
            alpha += rho * h_new
            h_prev = h_new

    return (W.detach().cpu().numpy() * eye_mask.cpu().numpy()).astype(np.float64)


def dagma_linear_fit(X: np.ndarray, cfg: Dict) -> np.ndarray:
    model = DagmaLinear(loss_type='l2')
    W = model.fit(X, lambda1=cfg["dagma_lambda1"])
    return np.asarray(W, dtype=np.float64)


def dagma_nonlinear_fit(X: np.ndarray, cfg: Dict) -> np.ndarray:
    d = X.shape[1]
    eq_model = DagmaMLP(dims=[d, int(cfg["dagma_hidden"]), 1], bias=True, dtype=torch.double)
    model = DagmaNonlinear(eq_model, dtype=torch.double)
    W = model.fit(X, lambda1=cfg["dagma_lambda1"], lambda2=cfg["dagma_lambda2"])
    return np.asarray(W, dtype=np.float64)


def causal_discovery(events_train_df: pd.DataFrame, q_train: np.ndarray, cfg: Dict) -> Dict:
    """Learn DAG on training split only; supports DAGMA and NOTEARS backends."""
    data = events_train_df.copy().astype(np.float64)
    data[TARGET_NAME] = np.asarray(q_train, dtype=np.float64)
    var_names = list(data.columns)

    X = data.values
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True) + 1e-9
    Xs = (X - X_mean) / X_std

    method_req = cfg.get("causal_method", "dagma_linear")
    method_used = method_req
    notes = []

    if method_req == "dagma_linear":
        if DAGMA_AVAILABLE:
            W = dagma_linear_fit(Xs, cfg)
        else:
            notes.append("dagma package not available; falling back to notears_linear")
            method_used = "notears_linear"
            W = notears_linear_torch(
                Xs,
                lambda1=cfg["notears_lambda1"],
                max_iter=cfg["notears_max_iter"],
                inner_steps=cfg["notears_inner_steps"],
                lr=cfg["notears_lr"],
                h_tol=cfg["notears_h_tol"],
                rho_max=cfg["notears_rho_max"],
                seed=SEED,
            )
    elif method_req == "dagma_nonlinear":
        if DAGMA_AVAILABLE:
            W = dagma_nonlinear_fit(Xs, cfg)
        else:
            notes.append("dagma package not available; falling back to notears_linear")
            method_used = "notears_linear"
            W = notears_linear_torch(
                Xs,
                lambda1=cfg["notears_lambda1"],
                max_iter=cfg["notears_max_iter"],
                inner_steps=cfg["notears_inner_steps"],
                lr=cfg["notears_lr"],
                h_tol=cfg["notears_h_tol"],
                rho_max=cfg["notears_rho_max"],
                seed=SEED,
            )
    else:
        method_used = "notears_linear"
        W = notears_linear_torch(
            Xs,
            lambda1=cfg["notears_lambda1"],
            max_iter=cfg["notears_max_iter"],
            inner_steps=cfg["notears_inner_steps"],
            lr=cfg["notears_lr"],
            h_tol=cfg["notears_h_tol"],
            rho_max=cfg["notears_rho_max"],
            seed=SEED,
        )

    threshold = cfg["causal_edge_threshold"]
    q_idx = var_names.index(TARGET_NAME)
    edge_list = []
    for i, src in enumerate(var_names):
        for j, dst in enumerate(var_names):
            if i == j:
                continue
            w = float(W[i, j])
            if abs(w) >= threshold:
                edge_list.append({"source": src, "target": dst, "weight": w, "abs_weight": abs(w)})
    edge_list = sorted(edge_list, key=lambda e: e["abs_weight"], reverse=True)

    q_parent_indices = [i for i in range(len(var_names)-1) if abs(W[i, q_idx]) >= threshold]
    q_parent_features = [var_names[i] for i in q_parent_indices]
    if len(q_parent_indices) == 0:
        q_parent_indices = list(range(len(FEATURE_COLUMNS)))
        q_parent_features = FEATURE_COLUMNS.copy()
        notes.append("No parent selected for q at threshold; fallback to all features")

    return {
        "variables": var_names,
        "weighted_adjacency": W,
        "edge_list": edge_list,
        "q_parent_indices": q_parent_indices,
        "q_parent_features": q_parent_features,
        "standardization": {"mean": X_mean.squeeze(0).tolist(), "std": X_std.squeeze(0).tolist()},
        "backend": {
            "requested": method_req,
            "used": method_used,
            "dagma_available": DAGMA_AVAILABLE,
            "notes": notes,
        },
    }


def plot_causal_heatmap(W: np.ndarray, labels: List[str], out_path: str):
    vmax = max(1e-3, np.max(np.abs(W)))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(W, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Child (target)")
    ax.set_ylabel("Parent (source)")
    ax.set_title("NOTEARS Weighted Adjacency (Training Data)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Edge weight")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def plot_causal_dag(W: np.ndarray, labels: List[str], out_path: str, threshold: float):
    d = len(labels)
    theta = np.linspace(0, 2 * np.pi, d, endpoint=False)
    xy = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_title("NOTEARS DAG (|weight| >= threshold)")
    ax.axis('off')

    for i, label in enumerate(labels):
        x, y = xy[i]
        ax.scatter(x, y, s=280, color='white', edgecolor='black', zorder=3)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, zorder=4)

    vmax = max(np.max(np.abs(W)), threshold + 1e-8)
    for i in range(d):
        for j in range(d):
            if i == j or abs(W[i, j]) < threshold:
                continue
            x0, y0 = xy[i]
            x1, y1 = xy[j]
            color = 'tab:red' if W[i, j] > 0 else 'tab:blue'
            width = 0.5 + 2.5 * (abs(W[i, j]) / vmax)
            ax.annotate(
                '',
                xy=(x1 * 0.92, y1 * 0.92),
                xytext=(x0 * 0.92, y0 * 0.92),
                arrowprops=dict(arrowstyle='->', lw=width, color=color, alpha=0.65),
                zorder=2,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


class SurrogateModel(nn.Module):
    def __init__(self, cfg, feature_mask=None):
        super().__init__()
        in_dim = len(FEATURE_COLUMNS)
        hidden = cfg["mlp_hidden"]
        layers = cfg["mlp_layers"]

        if feature_mask is None:
            feature_mask = np.ones(in_dim, dtype=np.float32)
        self.register_buffer("feature_mask", torch.tensor(feature_mask, dtype=torch.float32))

        blocks = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(p=cfg['dropout_p'])]
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(p=cfg['dropout_p'])])
        blocks.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*blocks)

    def forward(self, scalars):
        masked = scalars * self.feature_mask
        return self.net(masked)

def compute_eurotop_q_torch(scalars_raw):
    hs = scalars_raw[:, 1] + EPS
    tm = scalars_raw[:, 3] / 1.1 + EPS
    tan_alpha = scalars_raw[:, 12]
    rc = scalars_raw[:, 10]
    gamma_f = scalars_raw[:, 6]
    beta = scalars_raw[:, 0]

    gamma_b = torch.ones_like(hs)
    gamma_beta = torch.cos(deg2rad(beta.clamp(-80, 80)))
    gamma_v = torch.ones_like(hs)

    lm = G * tm**2 / (2 * math.pi)
    s = hs / lm
    xi = tan_alpha / torch.sqrt(s.clamp(min=EPS))

    TAN_ALPHA_VERTICAL_THRESHOLD = 0.05
    tan_alpha_safe = torch.clamp(tan_alpha, min=1e-4, max=10.0)

    q_non = 0.09 * torch.exp(
        - (1.5 * rc / (hs * gamma_f * gamma_beta + EPS)) ** 1.3
    )

    q_break = (0.023 / torch.sqrt(tan_alpha_safe)) * gamma_b * xi * torch.exp(
        - (2.7 * rc / (xi * hs * gamma_b * gamma_f * gamma_beta * gamma_v + EPS)) ** 1.3
    )

    use_only_non_breaking = (tan_alpha < TAN_ALPHA_VERTICAL_THRESHOLD)

    q_nd = torch.where(
        use_only_non_breaking,
        q_non,
        torch.minimum(q_break, q_non)
    )

    q_phys = q_nd * torch.sqrt(G * hs**3 + EPS)
    q_phys = torch.clamp(q_phys, min=0.0, max=1.0)

    return q_phys

def collate_fn(batch):
    scalars = torch.stack([b["scalars"] for b in batch], dim=0)
    scalars_raw = torch.stack([b["scalars_raw"] for b in batch], dim=0)
    target = torch.stack([b["target"] for b in batch], dim=0)
    return scalars, scalars_raw, target

def get_stage_weights(epoch, cfg):
    if epoch < cfg["warm_phys_start"]:
        return 0.0
    if epoch >= cfg["warm_phys_end"]:
        return 1.0
    t = (epoch - cfg["warm_phys_start"]) / max(1.0, (cfg["warm_phys_end"] - cfg["warm_phys_start"]))
    return t

def gaussian_nll_loss(mu, logvar, target):
    logvar = torch.clamp(logvar, min=-10.0, max=4.0)
    inv_var = torch.exp(-logvar)
    sq = (target - mu) ** 2
    nll = 0.5 * (logvar + sq * inv_var)
    
    reg = 0.005 * torch.mean(logvar ** 2)
    
    return nll.mean() + reg

def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def mc_sample_full(model, scalars, scalers, mc_samples=150, device=DEVICE):
    model.to(device)
    enable_mc_dropout(model)

    q_samples = []
    mu_log_list = []
    logvar_list = []

    with torch.no_grad():
        for _ in range(mc_samples):
            out = model(scalars)
            mu = out[:, 0]
            logvar = torch.clamp(out[:, 1], min=-8.0, max=6.0)
            std = torch.exp(0.5 * logvar)
            sample_log = mu + std * torch.randn_like(mu)
            q_sample = scalers['q'].inverse_torch(sample_log, device)
            q_sample = torch.clamp(q_sample, min=EPS, max=0.25)
            q_samples.append(q_sample)
            mu_log_list.append(mu)
            logvar_list.append(logvar)

    q_samples = torch.stack(q_samples, dim=0)
    total_std = q_samples.std(dim=0).cpu().numpy()

    mu_log = torch.stack(mu_log_list, dim=0)
    logvar = torch.stack(logvar_list, dim=0)

    epi_var = mu_log.var(dim=0)
    ale_var = torch.exp(logvar).mean(dim=0)

    epi_std_log = torch.sqrt(epi_var)
    ale_std_log = torch.sqrt(ale_var)

    mean_mu_log = mu_log.mean(dim=0)
    mean_q = scalers['q'].inverse_torch(mean_mu_log, device).cpu().numpy()

    epi_std = mean_q * epi_std_log.cpu().numpy()
    ale_std = mean_q * ale_std_log.cpu().numpy()

    return mean_q, total_std, epi_std, ale_std

def grad_norm(model):
    tot = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            tot += p.grad.data.norm(2).item() ** 2
            count += 1
    return math.sqrt(tot) if count > 0 else 0.0

def causal_gradient_penalty(q_mu, scalars_in, non_causal_idx):
    if len(non_causal_idx) == 0:
        return torch.tensor(0.0, device=scalars_in.device)
    grads = torch.autograd.grad(
        outputs=q_mu.sum(),
        inputs=scalars_in,
        create_graph=True,
        retain_graph=True,
    )[0]
    pen = torch.mean(torch.sum(grads[:, non_causal_idx] ** 2, dim=1))
    return pen

def train_epoch(model, loader, opt, scalers, epoch, cfg, non_causal_idx, wp: float):
    model.train()
    total_loss = total_data = total_phys = total_causal = total_grad = 0.0
    n = 0

    for scalars, scalars_raw, target in loader:
        scalars = scalars.to(DEVICE)
        scalars_raw = scalars_raw.to(DEVICE)
        target = target.to(DEVICE)

        need_input_grad = cfg["use_causal_grad_penalty"] and len(non_causal_idx) > 0
        if need_input_grad:
            scalars = scalars.detach().clone().requires_grad_(True)

        opt.zero_grad()
        out = model(scalars)
        q_mu = out[:, 0]
        q_logvar = out[:, 1]

        data_loss = cfg['w_q'] * gaussian_nll_loss(q_mu, q_logvar, target)
        q_pred_phys = scalers['q'].inverse_torch(q_mu, device=DEVICE)
        q_phys = compute_eurotop_q_torch(scalars_raw)
        phys_loss = wp * F.mse_loss(torch.log(q_pred_phys + EPS), torch.log(q_phys + EPS))

        causal_loss = torch.tensor(0.0, device=DEVICE)
        if need_input_grad:
            causal_loss = cfg["w_causal"] * causal_gradient_penalty(q_mu, scalars, non_causal_idx)

        loss = data_loss + phys_loss + causal_loss

        if not torch.isfinite(loss):
            print("Warning: non-finite loss; skipping step", loss.item())
            continue

        loss.backward()
        gn = grad_norm(model)
        opt.step()

        bs = scalars.shape[0]
        total_loss += float(loss.item()) * bs
        total_data += float(data_loss.item()) * bs
        total_phys += float(phys_loss.item()) * bs
        total_causal += float(causal_loss.item()) * bs
        total_grad += gn * bs
        n += bs

    denom = max(1, n)
    return (
        total_loss / denom,
        total_data / denom,
        total_phys / denom,
        total_causal / denom,
        total_grad / denom,
    )

def validate(model, val_loader, scalers, epoch, cfg, wp: float):
    model.eval()
    total_loss = total_data = total_phys = 0.0
    n = 0
    preds = {"q": []}
    trues = {"q": []}

    with torch.no_grad():
        for scalars, scalars_raw, target in val_loader:
            scalars = scalars.to(DEVICE)
            scalars_raw = scalars_raw.to(DEVICE)
            target = target.to(DEVICE)

            out = model(scalars)
            q_mu = out[:, 0]
            q_logvar = out[:, 1]
            data_loss = cfg['w_q'] * gaussian_nll_loss(q_mu, q_logvar, target)

            q_pred_phys = scalers['q'].inverse_torch(q_mu, device=DEVICE)
            q_phys = compute_eurotop_q_torch(scalars_raw)
            phys_loss = wp * F.mse_loss(torch.log(q_pred_phys + EPS), torch.log(q_phys + EPS))
            loss = data_loss + phys_loss

            bs = scalars.shape[0]
            total_loss += float(loss.item()) * bs
            total_data += float(data_loss.item()) * bs
            total_phys += float(phys_loss.item()) * bs
            n += bs

            q_np = scalers['q'].inverse_torch(q_mu, device=DEVICE).cpu().numpy()
            q_t = scalers['q'].inverse(target.cpu().numpy())
            preds["q"].extend(q_np)
            trues["q"].extend(q_t)

    denom = max(1, n)
    return total_loss / denom, total_data / denom, total_phys / denom, preds, trues

def get_predictions(model, loader, scalers, mc_samples=1):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for scalars, _, target in loader:
            scalars = scalars.to(DEVICE)

            if mc_samples > 1:
                mean_q, _, _, _ = mc_sample_full(model, scalars, scalers, mc_samples)
                q_pred = mean_q
            else:
                out = model(scalars)
                q_mu = out[:, 0]
                q_pred = scalers['q'].inverse_torch(q_mu, DEVICE).cpu().numpy()

            q_true = scalers['q'].inverse(target.numpy())

            preds.extend(q_pred)
            trues.extend(q_true)

    return np.array(preds), np.array(trues)

def get_predictions_with_uncertainty(model, loader, scalers, mc_samples=150):
    preds = []
    trues = []
    tot_all = []
    epi_all = []
    ale_all = []

    for scalars, _, target in loader:
        scalars = scalars.to(DEVICE)
        mean_q, total_std, epi_std, ale_std = mc_sample_full(model, scalars, scalers, mc_samples)
        q_true = scalers['q'].inverse(target.numpy())

        preds.extend(mean_q)
        trues.extend(q_true)
        tot_all.extend(total_std)
        epi_all.extend(epi_std)
        ale_all.extend(ale_std)

    return np.array(preds), np.array(trues), np.array(tot_all), np.array(epi_all), np.array(ale_all)

def compute_coverage(y_true, y_pred, total_std):
    lower = y_pred - 1.96 * total_std
    upper = y_pred + 1.96 * total_std
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    return coverage

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))

def compute_metrics(preds_dict, trues_dict):
    q_p = np.array(preds_dict["q"])
    q_t = np.array(trues_dict["q"])
    if q_p.size == 0:
        return {"q": {}}
    diff = q_p - q_t
    return {
        "q": {
            "MAE": float(np.mean(np.abs(diff))),
            "RMSE": float(np.sqrt(np.mean(diff ** 2))),
            "bias": float(np.mean(diff)),
            "R2": r2_score(q_t, q_p),
        }
    }

def build_feature_mask(parent_indices, use_mask):
    mask = np.ones(len(FEATURE_COLUMNS), dtype=np.float32)
    if use_mask:
        mask[:] = 0.0
        for idx in parent_indices:
            mask[idx] = 1.0
    return mask

def save_plot_data(data_dict: dict, filename: str):
    path = os.path.join(DATA_DIR, filename)
    try:
        df = pd.DataFrame(data_dict)
        df.to_excel(path, index=False)
        print(f"   → Plot data saved: {filename}")
    except Exception as e:
        print(f"Error saving plot data {filename}: {e}")

def plot_scatter_train_test(trues_train, preds_train, trues_test, preds_test, tot_test, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.3), sharex=True, sharey=True)

    ax1.scatter(trues_train, preds_train, s=48, alpha=0.78, color=COLOR_PREDICTED, marker='s',
                edgecolor='k', lw=0.65, label='Training', rasterized=True)
    ax1.plot([MIN_Q_PLOT, MAX_Q_PLOT], [MIN_Q_PLOT, MAX_Q_PLOT], 'k--', lw=1.6)

    mae = np.mean(np.abs(trues_train - preds_train))
    rmse = np.sqrt(np.mean((trues_train - preds_train)**2))
    r2 = r2_score(trues_train, preds_train)
    ax1.text(0.97, 0.03, f"MAE = {mae:.3e}\nRMSE = {rmse:.3e}\nR² = {r2:.4f}",
             transform=ax1.transAxes, va='bottom', ha='right', fontsize=9.5,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='0.7'))

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(MIN_Q_PLOT, MAX_Q_PLOT)
    ax1.set_ylim(MIN_Q_PLOT, MAX_Q_PLOT)
    ax1.set_xlabel('Observed q (m³/s/m)')
    ax1.set_ylabel('Predicted q (m³/s/m)')
    ax1.set_title('Training Set')
    ax1.legend(frameon=True, edgecolor='black', loc='upper left')
    ax1.grid(False)

    ax2.scatter(trues_test, preds_test, s=48, alpha=0.78, color=COLOR_TEST, marker='s',
                edgecolor='k', lw=0.65, label='Test', rasterized=True)
    ax2.plot([MIN_Q_PLOT, MAX_Q_PLOT], [MIN_Q_PLOT, MAX_Q_PLOT], 'k--', lw=1.6)

    sort_idx = np.argsort(trues_test)
    ax2.fill_between(trues_test[sort_idx], preds_test[sort_idx] - 1.96*tot_test[sort_idx],
                     preds_test[sort_idx] + 1.96*tot_test[sort_idx], color=COLOR_CI, alpha=0.32,
                     label='95% CI', zorder=1, rasterized=True)

    mae = np.mean(np.abs(trues_test - preds_test))
    rmse = np.sqrt(np.mean((trues_test - preds_test)**2))
    r2 = r2_score(trues_test, preds_test)
    ax2.text(0.97, 0.03, f"MAE = {mae:.3e}\nRMSE = {rmse:.3e}\nR² = {r2:.4f}",
             transform=ax2.transAxes, va='bottom', ha='right', fontsize=9.5,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='0.7'))

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(MIN_Q_PLOT, MAX_Q_PLOT)
    ax2.set_ylim(MIN_Q_PLOT, MAX_Q_PLOT)
    ax2.set_xlabel('Observed q (m³/s/m)')
    ax2.set_title('Test Set with 95% CI')
    ax2.legend(frameon=True, edgecolor='black', loc='upper left')
    ax2.grid(False)

    fig.suptitle('Observed vs Predicted Discharge', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    save_plot_data({
        'observed_train': trues_train,
        'predicted_train': preds_train,
        'observed_test': trues_test,
        'predicted_test': preds_test,
        'total_std_test': tot_test
    }, "data_scatter_train_test.xlsx")

def plot_scatter_obs_vs_pred_with_ci(trues, preds, stds, title, path, color, label, unc_type='Total', euro=None):
    fig, ax = plt.subplots(figsize=(6.2, 5.9))

    marker = 's'
    ax.scatter(trues, preds, s=48, alpha=0.78, color=color, marker=marker,
               edgecolor='k', lw=0.65, label=label, rasterized=True)

    if euro is not None:
        ax.scatter(trues, euro, s=52, alpha=0.72, color=COLOR_EUROTOP, marker='d',
                   edgecolor='k', lw=0.65, label='EurOtop', rasterized=True)

    ax.plot([MIN_Q_PLOT, MAX_Q_PLOT], [MIN_Q_PLOT, MAX_Q_PLOT], 'k--', lw=1.6, label='1:1')

    if np.any(stds > 0):
        sort_idx = np.argsort(trues)
        ax.fill_between(trues[sort_idx], preds[sort_idx] - 1.96*stds[sort_idx],
                        preds[sort_idx] + 1.96*stds[sort_idx], color=COLOR_CI, alpha=0.32,
                        label=f'95% CI ({unc_type})', zorder=1, rasterized=True)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(MIN_Q_PLOT, MAX_Q_PLOT)
    ax.set_ylim(MIN_Q_PLOT, MAX_Q_PLOT)
    ax.set_xlabel('Observed q (m³/s/m)')
    ax.set_ylabel('Predicted q (m³/s/m)')
    ax.set_title(title)

    mae = np.mean(np.abs(trues - preds))
    rmse = np.sqrt(np.mean((trues - preds)**2))
    r2 = r2_score(trues, preds)
    ax.text(0.97, 0.03, f"MAE = {mae:.3e}\nRMSE = {rmse:.3e}\nR² = {r2:.4f}",
            transform=ax.transAxes, va='bottom', ha='right', fontsize=9.5,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='0.7'))

    ax.legend(frameon=True, edgecolor='black', fontsize=10.5, loc='upper left')
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    data_dict = {
        'observed': trues,
        'predicted': preds,
    }
    if euro is not None:
        data_dict['eurotop'] = euro
    if np.any(stds > 0):
        data_dict['std'] = stds
    save_plot_data(data_dict, f"data_{os.path.basename(path).replace('.png','')}.xlsx")

def plot_scatter_obs_vs_euro(trues, euro, title, path, color=COLOR_EUROTOP):
    fig, ax = plt.subplots(figsize=(6.2, 5.9))

    ax.scatter(trues, euro, s=52, alpha=0.72, color=color, marker='d',
               edgecolor='k', lw=0.65, label='EurOtop', rasterized=True)
    ax.plot([MIN_Q_PLOT, MAX_Q_PLOT], [MIN_Q_PLOT, MAX_Q_PLOT], 'k--', lw=1.6, label='1:1')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(MIN_Q_PLOT, MAX_Q_PLOT)
    ax.set_ylim(MIN_Q_PLOT, MAX_Q_PLOT)
    ax.set_xlabel('Observed q (m³/s/m)')
    ax.set_ylabel('EurOtop q (m³/s/m)')
    ax.set_title(title)

    mae = np.mean(np.abs(trues - euro))
    rmse = np.sqrt(np.mean((trues - euro)**2))
    r2 = r2_score(trues, euro)
    ax.text(0.97, 0.03, f"MAE = {mae:.3e}\nRMSE = {rmse:.3e}\nR² = {r2:.4f}",
            transform=ax.transAxes, va='bottom', ha='right', fontsize=9.5,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='0.7'))

    ax.legend(frameon=True, edgecolor='black', fontsize=10.5, loc='upper left')
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    save_plot_data({
        'observed': trues,
        'eurotop': euro
    }, f"data_{os.path.basename(path).replace('.png','')}.xlsx")

def plot_uncertainty_vs_prediction(q_pred, unc_std, path, unc_type='Total', set_type='Test'):
    fig, ax = plt.subplots(figsize=(7.5, 6))

    color = COLOR_EPI if unc_type == 'Epistemic' else COLOR_ALE if unc_type == 'Aleatoric' else COLOR_TEST
    marker = 'o' if unc_type == 'Epistemic' else '^' if unc_type == 'Aleatoric' else 's'
    ax.scatter(q_pred, unc_std, s=36, alpha=0.70, color=color, marker=marker,
               edgecolor='k', lw=0.6, rasterized=True, label=f'{unc_type} uncertainty')

    minv = 5e-8
    maxv = 0.3
    ax.plot([minv, maxv], [minv, maxv], 'k--', lw=1.4, label='σ = q')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(minv, maxv)
    ax.set_ylim(minv, maxv)
    ax.set_xlabel('Predicted q (m³/s/m)')
    ax.set_ylabel(f'{unc_type} Uncertainty (Std)')
    ax.set_title(f'Predicted vs {unc_type} Uncertainty ({set_type})')
    ax.legend(frameon=True, edgecolor='black', loc='upper left', fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    save_plot_data({
        'predicted': q_pred,
        'uncertainty_std': unc_std
    }, f"data_{os.path.basename(path).replace('.png','')}.xlsx")

def plot_pred_vs_both_uncertainties(q_pred, epi_std, ale_std, path, set_type='Test'):
    fig, ax = plt.subplots(figsize=(7.5, 6))

    ax.scatter(q_pred, epi_std, s=36, alpha=0.70, color=COLOR_EPI, marker='o',
               edgecolor='k', lw=0.6, rasterized=True, label='Epistemic uncertainty')
    ax.scatter(q_pred, ale_std, s=36, alpha=0.70, color=COLOR_ALE, marker='^',
               edgecolor='k', lw=0.6, rasterized=True, label='Aleatoric uncertainty')

    minv = 5e-8
    maxv = 0.3
    ax.plot([minv, maxv], [minv, maxv], 'k--', lw=1.4, label='σ = q')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(minv, maxv)
    ax.set_ylim(minv, maxv)
    ax.set_xlabel('Predicted q (m³/s/m)')
    ax.set_ylabel('Uncertainty (Std)')
    ax.set_title(f'Predicted vs Epistemic & Aleatoric Uncertainty ({set_type})')

    mean_epi = np.mean(epi_std)
    mean_ale = np.mean(ale_std)
    stats = f"Mean Epistemic: {mean_epi:.2e}\nMean Aleatoric: {mean_ale:.2e}"
    ax.text(0.03, 0.03, stats, transform=ax.transAxes, va='bottom', ha='left', fontsize=9.5,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='0.7'))

    ax.legend(frameon=True, edgecolor='black', loc='upper left', fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    save_plot_data({
        'predicted': q_pred,
        'epistemic_std': epi_std,
        'aleatoric_std': ale_std
    }, f"data_{os.path.basename(path).replace('.png','')}.xlsx")

def plot_normalized_residuals_histogram(trues, preds, stds, path):
    residuals = trues - preds
    norm_res = residuals / (stds + 1e-10)

    fig, ax = plt.subplots(figsize=(7, 5.8))
    ax.hist(norm_res, bins=90, density=True, alpha=0.75, color='cornflowerblue', edgecolor='none')

    x = np.linspace(-8, 8, 500)
    ax.plot(x, 1/np.sqrt(2*np.pi) * np.exp(-0.5*x**2), 'r--', lw=2.4, label='N(0,1)')

    ax.set_xlim(-8, 8)
    ax.set_title('Normalized Residuals Distribution (Test)')
    ax.set_xlabel('Normalized Residual')
    ax.set_ylabel('Density')
    ax.legend(frameon=True, edgecolor='black', fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    save_plot_data({
        'normalized_residuals': norm_res
    }, f"data_{os.path.basename(path).replace('.png','')}.xlsx")

def plot_qq_normalized_residuals(trues, preds, stds, path):
    residuals = trues - preds
    norm_res = residuals / (stds + 1e-10)

    fig = plt.figure(figsize=(6.5, 6.5))
    probplot(norm_res, dist="norm", plot=plt)

    ax = plt.gca()
    ax.get_lines()[0].set_color('red')
    ax.get_lines()[0].set_linestyle('--')
    ax.get_lines()[0].set_linewidth(2.0)

    ax.get_lines()[1].set_marker('o')
    ax.get_lines()[1].set_markersize(4.5)
    ax.get_lines()[1].set_markeredgecolor('navy')
    ax.get_lines()[1].set_markerfacecolor('none')
    ax.get_lines()[1].set_markeredgewidth(1.0)

    ax.set_title('QQ-Plot of Normalized Residuals (Test)')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    save_plot_data({
        'normalized_residuals': norm_res
    }, f"data_{os.path.basename(path).replace('.png','')}.xlsx")

def plot_loss_history_option_A(history, path):
    if len(history.get("train_loss", [])) == 0:
        print("Warning: No loss history available. Skipping plot A.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 6))
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], label="Total Loss (Train)", color=COLOR_PREDICTED, lw=2.0)
    ax.plot(epochs, history["val_loss"], label="Total Loss (Val)", color='#d62728', ls='--', lw=2.0)
    ax.plot(epochs, history["train_data_loss"], label="Data Loss (Train)", color='#2ca02c', lw=1.5)
    ax.plot(epochs, history["train_phys_loss"], label="Physics Loss (Train)", color='#ff7f0e', lw=1.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Negative Log-Likelihood and Physics Regularization during Training')
    ax.legend(frameon=True, edgecolor='black', loc='upper right', fontsize=10, ncol=2)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    save_plot_data(history, "data_loss_history.xlsx")

def plot_q_vs_sample(obs, pred, euro, title, path, std=None, unc_type=None):
    sort_idx = np.argsort(obs)
    x = np.arange(len(obs))
    obs_s = obs[sort_idx]
    pred_s = pred[sort_idx]
    euro_s = euro[sort_idx]
    std_s = std[sort_idx] if std is not None else None

    fig, ax = plt.subplots(figsize=(9.4, 5.6))

    ax.scatter(x, obs_s,  marker='o', s=48, color=COLOR_OBSERVED, label='Observed', alpha=0.78, edgecolor='k', lw=0.65)
    ax.scatter(x, pred_s, marker='s', s=48, color=COLOR_PREDICTED, label='Predicted', alpha=0.78, edgecolor='k', lw=0.65)
    ax.scatter(x, euro_s, marker='d', s=52, color=COLOR_EUROTOP, label='EurOtop', alpha=0.72, edgecolor='k', lw=0.65)

    if std_s is not None:
        ax.fill_between(x, pred_s - 1.96*std_s, pred_s + 1.96*std_s,
                        color=COLOR_CI, alpha=0.32, label=f'95% CI ({unc_type})')

    ax.set_yscale('log')
    ax.set_ylim(MIN_Q_PLOT, MAX_Q_PLOT)
    ax.set_xlabel('Sample number (sorted by observed q)')
    ax.set_ylabel('q (m³/s/m)')
    ax.set_title(title)

    mae = np.mean(np.abs(obs_s - pred_s))
    rmse = np.sqrt(np.mean((obs_s - pred_s)**2))
    r2 = r2_score(obs_s, pred_s)
    ax.text(0.97, 0.03, f"MAE = {mae:.2e}\nRMSE = {rmse:.2e}\nR² = {r2:.3f}",
            transform=ax.transAxes, va='bottom', ha='right', fontsize=9.8,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='0.7'))

    ax.legend(loc='upper left', fontsize=9.8, framealpha=0.92)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    save_plot_data({
        'sample_idx': x,
        'observed': obs_s,
        'predicted': pred_s,
        'eurotop': euro_s,
        'std': std_s if std_s is not None else np.full_like(x, np.nan)
    }, f"data_{os.path.basename(path).replace('.png','')}.xlsx")

def plot_obs_vs_pred_sample_only(obs, pred, title, path):
    sort_idx = np.argsort(obs)
    x = np.arange(len(obs))
    obs_s = obs[sort_idx]
    pred_s = pred[sort_idx]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    ax.scatter(x, obs_s, marker='o', s=48, color=COLOR_OBSERVED, label='Observed', alpha=0.78, edgecolor='k', lw=0.65)
    ax.scatter(x, pred_s, marker='s', s=48, color=COLOR_PREDICTED, label='Predicted', alpha=0.78, edgecolor='k', lw=0.65)

    ax.set_yscale('log')
    ax.set_ylim(MIN_Q_PLOT, MAX_Q_PLOT)
    ax.set_xlabel('Sample number (sorted by observed q)')
    ax.set_ylabel('q (m³/s/m)')
    ax.set_title(title)

    mae  = np.mean(np.abs(obs_s - pred_s))
    rmse = np.sqrt(np.mean((obs_s - pred_s)**2))
    r2   = r2_score(obs_s, pred_s)
    stats = f"MAE = {mae:.3e}\nRMSE = {rmse:.3e}\nR² = {r2:.3f}"
    ax.text(0.97, 0.03, stats, transform=ax.transAxes, fontsize=10,
            va='bottom', ha='right',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='0.7'))

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=380, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    save_plot_data({
        'sample_idx': x,
        'observed': obs_s,
        'predicted': pred_s
    }, f"data_{os.path.basename(path).replace('.png','')}.xlsx")

def plot_q_vs_sample_slice(obs, pred, euro, std, title, path, start, end, unc_type=None):
    slice_obs = obs[start:end]
    slice_pred = pred[start:end]
    slice_euro = euro[start:end]
    slice_std = std[start:end] if std is not None else None

    plot_q_vs_sample(slice_obs, slice_pred, slice_euro, title, path, slice_std, unc_type)

def plot_std_vs_sample_sorted(q_pred, std, title, path, unc_type='Total'):
    sort_idx = np.argsort(q_pred)
    x = np.arange(len(q_pred))
    std_s = std[sort_idx]

    fig, ax = plt.subplots(figsize=(9.4, 5.6))

    color = COLOR_EPI if unc_type == 'Epistemic' else COLOR_ALE if unc_type == 'Aleatoric' else COLOR_TEST
    ax.scatter(x, std_s, marker='o', s=48, color=color, label=f'{unc_type} Std', alpha=0.78, edgecolor='k', lw=0.65)

    ax.set_yscale('log')
    ax.set_ylim(1e-7, 0.1)
    ax.set_xlabel('Sample number (sorted by predicted q)')
    ax.set_ylabel('Uncertainty Std (m³/s/m)')
    ax.set_title(title)

    mean_std = np.mean(std_s)
    median_std = np.median(std_s)
    stats = f"Mean Std = {mean_std:.2e}\nMedian Std = {median_std:.2e}"
    ax.text(0.97, 0.03, stats, transform=ax.transAxes, fontsize=9.8,
            va='bottom', ha='right',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='0.7'))

    ax.legend(loc='upper left', fontsize=9.8, framealpha=0.92)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    save_plot_data({
        'sample_idx': x,
        'predicted_q': q_pred[sort_idx],
        'uncertainty_std': std_s
    }, f"data_{os.path.basename(path).replace('.png','')}.xlsx")

def plot_differences_vs_sample(obs, pred, euro, title, path):
    sort_idx = np.argsort(obs)
    x = np.arange(len(obs))
    diff_pred = obs[sort_idx] - pred[sort_idx]
    diff_euro = obs[sort_idx] - euro[sort_idx]

    fig, ax = plt.subplots(figsize=(9.4, 5.6))

    ax.scatter(x, diff_pred, marker='s', s=48, color=COLOR_PREDICTED, label='Observed - Predicted', alpha=0.78, edgecolor='k', lw=0.65)
    ax.scatter(x, diff_euro, marker='d', s=52, color=COLOR_EUROTOP, label='Observed - EurOtop', alpha=0.72, edgecolor='k', lw=0.65)

    ax.set_xlabel('Sample number (sorted by observed q)')
    ax.set_ylabel('Difference in q (m³/s/m)')
    ax.set_title(title)

    ax.text(0.97, 0.97,
            f"Obs-Pred:  mean = {np.mean(diff_pred):.2e}   std = {np.std(diff_pred):.2e}\n"
            f"Obs-Euro: mean = {np.mean(diff_euro):.2e}   std = {np.std(diff_euro):.2e}",
            transform=ax.transAxes, va='top', ha='right', fontsize=9.8,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='0.7'))

    ax.legend(loc='upper left', fontsize=9.8, framealpha=0.92)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    save_plot_data({
        'sample_idx': x,
        'obs_minus_pred': diff_pred,
        'obs_minus_euro': diff_euro
    }, f"data_{os.path.basename(path).replace('.png','')}.xlsx")

def save_excel_with_details(events_df, q_obs, q_pred, q_euro, filename):
    full_path = os.path.join(ROOT_OUT, filename)
    try:
        df_out = events_df.copy().reset_index(drop=True)
        df_out['q_observed']  = q_obs
        df_out['q_predicted'] = q_pred
        df_out['q_eurotop']   = q_euro
        cols = list(events_df.columns) + ['q_observed', 'q_predicted', 'q_eurotop']
        df_out = df_out[cols]
        df_out.to_excel(full_path, index=False)
        print("Excel saved successfully:", full_path)
    except PermissionError:
        print("\n" + "="*80)
        print("PERMISSION ERROR - cannot write file!")
        print(f"File path: {full_path}")
        print("احتمالاً فایل اکسل باز است. همه پنجره‌های اکسل را ببندید.")
        print("یا فایل را دستی تغییر نام دهید / حذف کنید.")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        raise

def main():
    clash_path = r"CLASH database.csv"
    events, q_arr = parse_clash(clash_path)
    n_total = len(q_arr)
    print("Loaded CLASH records:", n_total)

    idx = np.random.permutation(n_total)
    ntr = max(1, int(0.7 * n_total))
    nval = max(1, int(0.15 * n_total))
    nte = n_total - ntr - nval
    tr_idx = idx[:ntr]
    va_idx = idx[ntr:ntr + nval]
    te_idx = idx[ntr + nval:]
    print("Split (train,val,test):", len(tr_idx), len(va_idx), len(te_idx))

    tr_events = events.iloc[tr_idx].reset_index(drop=True)
    va_events = events.iloc[va_idx].reset_index(drop=True)
    te_events = events.iloc[te_idx].reset_index(drop=True)
    tr_q, va_q, te_q = q_arr[tr_idx], q_arr[va_idx], q_arr[te_idx]

    causal = causal_discovery_notears(tr_events[FEATURE_COLUMNS], tr_q, CFG)
    W = causal["weighted_adjacency"]
    np.save(os.path.join(CAUSAL_DIR, "notears_weighted_adjacency.npy"), W)
    with open(os.path.join(CAUSAL_DIR, "notears_edges.json"), "w") as f:
        json.dump({
            "variables": causal["variables"],
            "q_parent_features": causal["q_parent_features"],
            "edge_threshold": CFG["causal_edge_threshold"],
            "edges": causal["edge_list"],
        }, f, indent=2)
    plot_causal_heatmap(W, causal["variables"], os.path.join(CAUSAL_DIR, "notears_heatmap.png"))
    plot_causal_dag(W, causal["variables"], os.path.join(CAUSAL_DIR, "notears_dag.png"), CFG["causal_edge_threshold"])
    print("Causal results saved to", CAUSAL_DIR)
    print("Parents of q:", causal["q_parent_features"])

    parent_idx = causal["q_parent_indices"]
    non_causal_idx = [i for i in range(len(FEATURE_COLUMNS)) if i not in set(parent_idx)]
    feature_mask = build_feature_mask(parent_idx, CFG["use_causal_mask"])

    scalers = {
        'q': Scaler(log=True),
    }
    scalers['q'].fit(tr_q)
    scalar_arr = tr_events.values.astype(np.float32)
    scalers['scalars'] = {
        'mean': np.mean(scalar_arr, axis=0).astype(np.float32),
        'std': np.std(scalar_arr, axis=0).astype(np.float32) + 1e-9,
    }

    import pickle
    with open(os.path.join(CKPT_DIR, "scaler_overtopping.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    train_ds = OvertoppingSurrogateDataset(tr_events, tr_q, scalers)
    val_ds = OvertoppingSurrogateDataset(va_events, va_q, scalers)
    test_ds = OvertoppingSurrogateDataset(te_events, te_q, scalers)
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False, collate_fn=collate_fn)

    model = SurrogateModel(CFG, feature_mask=feature_mask).to(DEVICE)

    w_phys_log = nn.Parameter(torch.log(torch.tensor(1e-2, dtype=torch.float32)))

    opt = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': CFG['lr']},
        {'params': [w_phys_log], 'lr': CFG['lr'] * 0.05}
    ], weight_decay=CFG['weight_decay'])

    history = {
        "train_loss": [], "train_data_loss": [], "train_phys_loss": [],
        "train_causal_loss": [], "train_grad_norm": [],
        "val_loss": [], "val_data_loss": [], "val_phys_loss": [],
        "w_phys_history": []
    }

    best_val_r2 = -np.inf
    best_epoch = 0
    epochs_no_improve = 0

    for ep in range(1, CFG['epochs'] + 1):
        t0 = time.time()

        warm_factor = get_stage_weights(ep, CFG)
        w_phys_value = math.exp(w_phys_log.item()) * warm_factor
        wp = w_phys_value

        tr_loss, tr_data, tr_phys, tr_causal, tr_grad = train_epoch(
            model, train_loader, opt, scalers, ep, CFG, non_causal_idx, wp
        )
        va_loss, va_data, va_phys, _, _ = validate(model, val_loader, scalers, ep, CFG, wp)

        history["train_loss"].append(tr_loss)
        history["train_data_loss"].append(tr_data)
        history["train_phys_loss"].append(tr_phys)
        history["train_causal_loss"].append(tr_causal)
        history["train_grad_norm"].append(tr_grad)
        history["val_loss"].append(va_loss)
        history["val_data_loss"].append(va_data)
        history["val_phys_loss"].append(va_phys)
        history["w_phys_history"].append(math.exp(w_phys_log.item()))

        va_preds, va_trues = get_predictions(model, val_loader, scalers, mc_samples=0)
        va_r2 = r2_score(va_trues, va_preds)

        if ep % CFG['print_every'] == 0:
            dt = time.time() - t0
            print(f"Epoch {ep:03d}  train_loss={tr_loss:.6f}  val_loss={va_loss:.6f}  "
                  f"val_r2={va_r2:.4f}  w_phys={math.exp(w_phys_log.item()):.2e}  time={dt:.1f}s")

        improved = va_r2 > best_val_r2 + 1e-5

        if improved:
            best_val_r2 = va_r2
            best_epoch = ep
            epochs_no_improve = 0

            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "w_phys_log": w_phys_log,
                "val_loss": va_loss,
                "val_r2": va_r2,
            }, os.path.join(CKPT_DIR, "best_model.pth"))

            print(f"New best model saved at epoch {ep}  (val_r2={va_r2:.4f}, w_phys={math.exp(w_phys_log.item()):.2e})")

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CFG['patience']:
                print(f"Early stopping at epoch {ep}")
                print(f"Best epoch was {best_epoch} with val_r2={best_val_r2:.4f}")
                break

    torch.save({
        "model_state_dict": model.state_dict(),
        "w_phys_log": w_phys_log,
    }, os.path.join(CKPT_DIR, "final_model.pth"))

    checkpoint = torch.load(os.path.join(CKPT_DIR, "best_model.pth"), map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "w_phys_log" in checkpoint:
        w_phys_log.data = checkpoint["w_phys_log"]
    model.to(DEVICE)

    print(f"Loaded best model from epoch {checkpoint['epoch']}  val_r2={checkpoint.get('val_r2', 'N/A'):.4f}")
    print(f"Learned physics weight: {math.exp(w_phys_log.item()):.2e}")

    preds_train, trues_train, tot_train, epi_train, ale_train = get_predictions_with_uncertainty(
        model, train_loader, scalers, CFG['mc_samples']
    )

    preds_test, trues_test, tot_test, epi_test, ale_test = get_predictions_with_uncertainty(
        model, test_loader, scalers, CFG['mc_samples']
    )

    q_eurotop_test = []
    with torch.no_grad():
        for _, scalars_raw, _ in test_loader:
            scalars_raw = scalars_raw.to(DEVICE)
            q_phys = compute_eurotop_q_torch(scalars_raw).cpu().numpy()
            q_eurotop_test.extend(q_phys)
    q_eurotop_test = np.array(q_eurotop_test)

    q_eurotop_train = []
    with torch.no_grad():
        for _, scalars_raw, _ in train_loader:
            scalars_raw = scalars_raw.to(DEVICE)
            q_phys = compute_eurotop_q_torch(scalars_raw).cpu().numpy()
            q_eurotop_train.extend(q_phys)
    q_eurotop_train = np.array(q_eurotop_train)

    print(f"Average total std (test): {np.mean(tot_test):.2e}")

    save_excel_with_details(tr_events, trues_train, preds_train, q_eurotop_train, "predictions_train.xlsx")
    save_excel_with_details(te_events, trues_test, preds_test, q_eurotop_test, "predictions_test.xlsx")

    train_metrics = compute_metrics({"q": preds_train}, {"q": trues_train})
    test_metrics = compute_metrics({"q": preds_test}, {"q": trues_test})

    coverage_95 = compute_coverage(trues_test, preds_test, tot_test)

    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump({
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "coverage_95": float(coverage_95),
            "mean_total_std_test": float(np.mean(tot_test)),
            "final_w_phys": float(math.exp(w_phys_log.item())),
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
            "causal_q_parents": causal["q_parent_features"],
        }, f, indent=2, default=float)

    print("TRAIN METRICS:")
    print(json.dumps(train_metrics, indent=2, default=float))
    print("TEST METRICS:")
    print(json.dumps(test_metrics, indent=2, default=float))
    print(f"95% coverage on test set: {coverage_95:.4f}")

    print("\nGenerating plots and saving raw data...")

    plot_scatter_train_test(
        trues_train, preds_train, trues_test, preds_test, tot_test,
        os.path.join(PLOTS_DIR, "scatter_observed_vs_predicted_with_CI.png")
    )

    plot_scatter_obs_vs_pred_with_ci(
        trues_train, preds_train, np.zeros_like(preds_train),
        'Train: Observed vs Predicted',
        os.path.join(PLOTS_DIR, "obs_vs_pred_train_no_ci.png"),
        color=COLOR_PREDICTED, label='Train', unc_type='None'
    )

    plot_scatter_obs_vs_pred_with_ci(
        trues_test, preds_test, np.zeros_like(preds_test),
        'Test: Observed vs Predicted',
        os.path.join(PLOTS_DIR, "obs_vs_pred_test_no_ci.png"),
        color=COLOR_TEST, label='Test', unc_type='None'
    )

    plot_scatter_obs_vs_euro(
        trues_train, q_eurotop_train,
        'Train: Observed vs EurOtop',
        os.path.join(PLOTS_DIR, "obs_vs_euro_train.png")
    )

    plot_scatter_obs_vs_euro(
        trues_test, q_eurotop_test,
        'Test: Observed vs EurOtop',
        os.path.join(PLOTS_DIR, "obs_vs_euro_test.png")
    )

    plot_uncertainty_vs_prediction(
        preds_test, tot_test,
        os.path.join(PLOTS_DIR, "uncertainty_vs_prediction_total_test.png"),
        'Total', 'Test'
    )

    plot_uncertainty_vs_prediction(
        preds_test, epi_test,
        os.path.join(PLOTS_DIR, "uncertainty_vs_prediction_epistemic_test.png"),
        'Epistemic', 'Test'
    )

    plot_uncertainty_vs_prediction(
        preds_test, ale_test,
        os.path.join(PLOTS_DIR, "uncertainty_vs_prediction_aleatoric_test.png"),
        'Aleatoric', 'Test'
    )

    plot_uncertainty_vs_prediction(
        preds_train, tot_train,
        os.path.join(PLOTS_DIR, "uncertainty_vs_prediction_total_train.png"),
        'Total', 'Train'
    )

    plot_uncertainty_vs_prediction(
        preds_train, epi_train,
        os.path.join(PLOTS_DIR, "uncertainty_vs_prediction_epistemic_train.png"),
        'Epistemic', 'Train'
    )

    plot_uncertainty_vs_prediction(
        preds_train, ale_train,
        os.path.join(PLOTS_DIR, "uncertainty_vs_prediction_aleatoric_train.png"),
        'Aleatoric', 'Train'
    )

    plot_scatter_obs_vs_pred_with_ci(
        trues_test, preds_test, tot_test,
        'Test: Observed vs Predicted with Total Uncertainty',
        os.path.join(PLOTS_DIR, "obs_vs_pred_with_total_unc_test.png"),
        color=COLOR_TEST, label='Test', unc_type='Total'
    )

    plot_scatter_obs_vs_pred_with_ci(
        trues_test, preds_test, epi_test,
        'Test: Observed vs Predicted with Epistemic Uncertainty',
        os.path.join(PLOTS_DIR, "obs_vs_pred_with_epi_unc_test.png"),
        color=COLOR_TEST, label='Test', unc_type='Epistemic'
    )

    plot_scatter_obs_vs_pred_with_ci(
        trues_test, preds_test, ale_test,
        'Test: Observed vs Predicted with Aleatoric Uncertainty',
        os.path.join(PLOTS_DIR, "obs_vs_pred_with_ale_unc_test.png"),
        color=COLOR_TEST, label='Test', unc_type='Aleatoric'
    )

    plot_scatter_obs_vs_pred_with_ci(
        trues_train, preds_train, tot_train,
        'Train: Observed vs Predicted with Total Uncertainty',
        os.path.join(PLOTS_DIR, "obs_vs_pred_with_total_unc_train.png"),
        color=COLOR_PREDICTED, label='Train', unc_type='Total'
    )

    plot_scatter_obs_vs_pred_with_ci(
        trues_train, preds_train, epi_train,
        'Train: Observed vs Predicted with Epistemic Uncertainty',
        os.path.join(PLOTS_DIR, "obs_vs_pred_with_epi_unc_train.png"),
        color=COLOR_PREDICTED, label='Train', unc_type='Epistemic'
    )

    plot_scatter_obs_vs_pred_with_ci(
        trues_train, preds_train, ale_train,
        'Train: Observed vs Predicted with Aleatoric Uncertainty',
        os.path.join(PLOTS_DIR, "obs_vs_pred_with_ale_unc_train.png"),
        color=COLOR_PREDICTED, label='Train', unc_type='Aleatoric'
    )

    plot_q_vs_sample(
        trues_train, preds_train, q_eurotop_train,
        "Train: Observed vs Predicted vs EurOtop",
        os.path.join(PLOTS_DIR, "q_vs_sample_train.png")
    )

    plot_q_vs_sample(
        trues_test, preds_test, q_eurotop_test,
        "Test: Observed vs Predicted vs EurOtop",
        os.path.join(PLOTS_DIR, "q_vs_sample_test.png")
    )

    plot_obs_vs_pred_sample_only(
        trues_train, preds_train,
        "Train: Observed vs Predicted (by sample)",
        os.path.join(PLOTS_DIR, "obs_vs_pred_train_by_sample.png")
    )

    plot_obs_vs_pred_sample_only(
        trues_test, preds_test,
        "Test: Observed vs Predicted (by sample)",
        os.path.join(PLOTS_DIR, "obs_vs_pred_test_by_sample.png")
    )

    slice_size = 50
    for i in range(3):
        start = i * slice_size
        end = start + slice_size
        if end > len(trues_test):
            break
        plot_q_vs_sample_slice(
            trues_test, preds_test, q_eurotop_test, tot_test,
            f"Test Set Samples {start}-{end}",
            os.path.join(PLOTS_DIR, f"q_vs_sample_test_slice_{start}_{end}.png"),
            start, end, 'Total'
        )

    for i in range(3):
        start = i * slice_size
        end = start + slice_size
        if end > len(trues_train):
            break
        plot_q_vs_sample_slice(
            trues_train, preds_train, q_eurotop_train, tot_train,
            f"Train Set Samples {start}-{end}",
            os.path.join(PLOTS_DIR, f"q_vs_sample_train_slice_{start}_{end}.png"),
            start, end, 'Total'
        )

    plot_q_vs_sample(
        trues_train, preds_train, q_eurotop_train,
        "Train: Observed vs Predicted vs EurOtop with Total CI",
        os.path.join(PLOTS_DIR, "q_vs_sample_train_with_total_ci.png"),
        tot_train, 'Total'
    )

    plot_q_vs_sample(
        trues_test, preds_test, q_eurotop_test,
        "Test: Observed vs Predicted vs EurOtop with Total CI",
        os.path.join(PLOTS_DIR, "q_vs_sample_test_with_total_ci.png"),
        tot_test, 'Total'
    )

    plot_normalized_residuals_histogram(
        trues_test, preds_test, tot_test,
        os.path.join(PLOTS_DIR, "normalized_residuals_histogram.png")
    )

    plot_qq_normalized_residuals(
        trues_test, preds_test, tot_test,
        os.path.join(PLOTS_DIR, "qq_plot_normalized_residuals.png")
    )

    if len(history.get("train_loss", [])) > 0:
        plot_loss_history_option_A(
            history,
            os.path.join(PLOTS_DIR, "loss_history_option_A_NLL_title.png")
        )

    plot_scatter_obs_vs_pred_with_ci(
        trues_train, preds_train, tot_train,
        'Train: Observed vs Predicted vs EurOtop with Total CI',
        os.path.join(PLOTS_DIR, "obs_vs_pred_vs_euro_with_ci_train.png"),
        color=COLOR_PREDICTED, label='Predicted', unc_type='Total', euro=q_eurotop_train
    )

    plot_scatter_obs_vs_pred_with_ci(
        trues_test, preds_test, tot_test,
        'Test: Observed vs Predicted vs EurOtop with Total CI',
        os.path.join(PLOTS_DIR, "obs_vs_pred_vs_euro_with_ci_test.png"),
        color=COLOR_TEST, label='Predicted', unc_type='Total', euro=q_eurotop_test
    )

    plot_scatter_obs_vs_pred_with_ci(
        trues_train, preds_train, np.zeros_like(tot_train),
        'Train: Observed vs Predicted vs EurOtop',
        os.path.join(PLOTS_DIR, "obs_vs_pred_vs_euro_train.png"),
        color=COLOR_PREDICTED, label='Predicted', unc_type=None, euro=q_eurotop_train
    )

    plot_scatter_obs_vs_pred_with_ci(
        trues_test, preds_test, np.zeros_like(tot_test),
        'Test: Observed vs Predicted vs EurOtop',
        os.path.join(PLOTS_DIR, "obs_vs_pred_vs_euro_test.png"),
        color=COLOR_TEST, label='Predicted', unc_type=None, euro=q_eurotop_test
    )


    plot_pred_vs_both_uncertainties(
        preds_test, epi_test, ale_test,
        os.path.join(PLOTS_DIR, "pred_vs_both_unc_test.png"),
        'Test'
    )

    plot_pred_vs_both_uncertainties(
        preds_train, epi_train, ale_train,
        os.path.join(PLOTS_DIR, "pred_vs_both_unc_train.png"),
        'Train'
    )

    plot_std_vs_sample_sorted(
        preds_test, tot_test,
        "Test: Total Std (sorted)",
        os.path.join(PLOTS_DIR, "std_vs_sample_test_total.png"),
        'Total'
    )

    plot_std_vs_sample_sorted(
        preds_test, epi_test,
        "Test: Epistemic Std (sorted)",
        os.path.join(PLOTS_DIR, "std_vs_sample_test_epi.png"),
        'Epistemic'
    )

    plot_std_vs_sample_sorted(
        preds_test, ale_test,
        "Test: Aleatoric Std (sorted)",
        os.path.join(PLOTS_DIR, "std_vs_sample_test_ale.png"),
        'Aleatoric'
    )

    plot_std_vs_sample_sorted(
        preds_train, tot_train,
        "Train: Total Std (sorted)",
        os.path.join(PLOTS_DIR, "std_vs_sample_train_total.png"),
        'Total'
    )

    plot_std_vs_sample_sorted(
        preds_train, epi_train,
        "Train: Epistemic Std (sorted)",
        os.path.join(PLOTS_DIR, "std_vs_sample_train_epi.png"),
        'Epistemic'
    )

    plot_std_vs_sample_sorted(
        preds_train, ale_train,
        "Train: Aleatoric Std (sorted)",
        os.path.join(PLOTS_DIR, "std_vs_sample_train_ale.png"),
        'Aleatoric'
    )

    plot_differences_vs_sample(
        trues_test, preds_test, q_eurotop_test,
        "Test: Differences vs Sample (sorted)",
        os.path.join(PLOTS_DIR, "differences_vs_sample_test.png")
    )

    plot_differences_vs_sample(
        trues_train, preds_train, q_eurotop_train,
        "Train: Differences vs Sample (sorted)",
        os.path.join(PLOTS_DIR, "differences_vs_sample_train.png")
    )

    print("\nAll plots and corresponding raw data files have been saved.")
    print(f"Plots directory     : {PLOTS_DIR}")
    print(f"Raw data directory  : {DATA_DIR}")
    print("Finished.")

if __name__ == "__main__":
    main()
