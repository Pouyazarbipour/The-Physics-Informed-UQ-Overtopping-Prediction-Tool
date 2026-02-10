"""
Staged physics surrogate with aleatoric + epistemic uncertainty (MC Dropout) for overtopping prediction
No Excel outputs — only .npz/.npy exports for q.
Uses CLASH database from CSV and EuroTop 2018 formula for physics loss.
Updated for journal-ready figures, R², train/test separation, logarithmic scatter plots.
"""
import os, math, time, random, json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set matplotlib for journal-ready
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.5, 3.5),
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ----------------- CONFIG -----------------
ROOT_OUT = os.path.join(os.getcwd(), "surr_outputs_overtopping_phys_epistemic_no_excel")
PLOTS_DIR = os.path.join(ROOT_OUT, "plots")
REPORTS_DIR = os.path.join(ROOT_OUT, "reports")
CKPT_DIR = os.path.join(ROOT_OUT, "checkpoints")
os.makedirs(ROOT_OUT, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
CFG = {
    "lr": 3e-4,
    "batch_size": 16,
    "epochs": 400,
    "mlp_hidden": 512,
    "mlp_layers": 4,
    "weight_decay": 1e-6,
    "patience": 30,
    "print_every": 1,
    # data loss weights
    "w_q": 1.0,
    # physics base weights (small)
    "w_phys_base": 1e-3,
    # warmup schedule (epochs)
    "warm_phys_start": 10,
    "warm_phys_end": 40,
    # MC dropout samples for epistemic uncertainty (user requested 50)
    "mc_samples": 150,
    # dropout prob used in model
    "dropout_p": 0.12,
}
G = 9.81
EPS = 1e-8
# ---------- Utilities ----------
def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi
def clamp_tensor(a, minv=1e-8, maxv=1e8):
    return torch.clamp(a, min=minv, max=maxv)
# ---------- Scaler ----------
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
# ---------- Readers ----------
def parse_clash(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    required = ['q', 'beta', 'hm0', 'h', 'tm-10_toe', 'bt', 'ht', 'gama_f', 'cotad', 'cotau', 'hb', 'rc', 'b', 'tanb', 'ac', 'gc']
    for r in required:
        if r not in df.columns:
            raise RuntimeError(f"CLASH CSV missing column: {r}")
    events = df[required].copy()  # include all for events
    q_values = events["q"].values.astype(np.float32)
    events = events.drop(columns=['q'])  # remove q from events
    return events.reset_index(drop=True), q_values
# ---------- Dataset ----------
class OvertoppingSurrogateDataset(Dataset):
    def __init__(self, events_df, q_arr, scalers):
        self.events = events_df.reset_index(drop=True)
        self.q = q_arr.astype(np.float32)
        self.scalers = scalers
    def __len__(self):
        return len(self.q)
    def __getitem__(self, idx):
        row = self.events.iloc[idx]
        scalars_raw = np.array([
            float(row["beta"]),
            float(row["hm0"]),
            float(row["h"]),
            float(row["tm-10_toe"]),
            float(row["bt"]),
            float(row["ht"]),
            float(row["gama_f"]),
            float(row["cotad"]),
            float(row["cotau"]),
            float(row["hb"]),
            float(row["rc"]),
            float(row["b"]),
            float(row["tanb"]),
            float(row["ac"]),
            float(row["gc"])], dtype=np.float32)
        scalars = (scalars_raw - self.scalers['scalars']['mean']) / (self.scalers['scalars']['std'] + 1e-9)
        q_t = self.scalers['q'].transform(self.q[idx])
        return {"scalars": torch.from_numpy(scalars), "scalars_raw": torch.from_numpy(scalars_raw), "target": torch.tensor(q_t)}
# ---------- Model (MLP with Dropout) ----------
class SurrogateModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = 15
        hidden = cfg["mlp_hidden"]
        layers = cfg["mlp_layers"]
        self.layers = []
        self.layers.append(nn.Linear(in_dim, hidden))
        self.layers.append(nn.SiLU())
        self.layers.append(nn.Dropout(p=cfg['dropout_p']))
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.SiLU())
            self.layers.append(nn.Dropout(p=cfg['dropout_p']))
        self.layers.append(nn.Linear(hidden, 2)) # mu, logvar for q
        self.net = nn.Sequential(*self.layers)
    def forward(self, scalars):
        out = self.net(scalars)
        return out
# ---------- Physics solver (EuroTop 2018) ----------
def compute_eurotop_q_torch(scalars_raw):
    # scalars_raw: B x 15
    # indices: 0:beta, 1:hm0, 2:h, 3:tm-10_toe, 4:bt, 5:ht, 6:gama_f, 7:cotad, 8:cotau, 9:hb, 10:rc, 11:b, 12:tanb, 13:ac, 14:gc
    hs = scalars_raw[:, 1] + EPS
    tm = scalars_raw[:, 3] / 1.1 + EPS  # approximate Tm-1,0
    tan_alpha = scalars_raw[:, 12] + EPS
    rc = scalars_raw[:, 10]
    gamma_f = scalars_raw[:, 6]
    beta = scalars_raw[:, 0]
    gamma_b = torch.ones_like(hs) # assume no berm
    gamma_beta = torch.cos(deg2rad(beta.clamp(-80,80))) # approximate
    gamma_v = torch.ones_like(hs) # assume no vertical wall influence
    lm = G * tm**2 / (2 * math.pi)
    s = hs / lm
    xi = tan_alpha / torch.sqrt(s.clamp(min=EPS))
    # Q for breaking waves
    Q_break = (0.023 / torch.sqrt(tan_alpha)) * gamma_b * xi * torch.exp( - (2.7 * rc / (xi * hs * gamma_b * gamma_f * gamma_beta * gamma_v )) ** 1.3 )
    # Q for non-breaking waves
    Q_non = 0.09 * torch.exp( - (1.5 * rc / (hs * gamma_f * gamma_beta)) ** 1.3 )
    # Take minimum
    Q = torch.minimum(Q_break, Q_non)
    # Dimensional q
    q_phys = Q * torch.sqrt(G * hs**3)
    return q_phys
# ---------- Training / Utils ----------
def collate_fn(batch):
    scalars = torch.stack([b["scalars"] for b in batch], dim=0)
    scalars_raw = torch.stack([b["scalars_raw"] for b in batch], dim=0)
    target = torch.stack([b["target"] for b in batch], dim=0)
    return scalars, scalars_raw, target
def get_stage_weights(epoch, cfg):
    wp = 0.0
    if epoch >= cfg["warm_phys_start"]:
        if epoch >= cfg["warm_phys_end"]:
            wp = cfg["w_phys_base"]
        else:
            t = (epoch - cfg["warm_phys_start"]) / max(1.0, (cfg["warm_phys_end"] - cfg["warm_phys_start"]))
            wp = cfg["w_phys_base"] * t
    return wp
# Gaussian heteroscedastic NLL (uses log-variance predicted)
def gaussian_nll_loss(mu, logvar, target, reduce_mean=True):
    logvar = torch.clamp(logvar, min=-20.0, max=8.0)
    inv_var = torch.exp(-logvar)
    sq = (target - mu)**2
    loss = 0.5 * (logvar + sq * inv_var)
    if reduce_mean:
        return loss.mean()
    else:
        return loss
# helper: enable dropout for MC (keeps batchnorm in eval)
def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
# MC predict returning mean and epistemic variance + aleatoric mean
def mc_predict(model, scalars, scalars_raw, scalers, mc_samples=50, device=DEVICE):
    model.to(device)
    enable_mc_dropout(model)
    preds = []
    aleatoric_vars = []
    with torch.no_grad():
        for i in range(mc_samples):
            out = model(scalars) # B x 2
            q_mu = out[:, 0]
            q_logvar = out[:, 1]
            preds.append(q_mu)
            aleatoric_vars.append(torch.exp(q_logvar))
    preds = torch.stack(preds, dim=0)
    aleatoric_vars = torch.stack(aleatoric_vars, dim=0)
    pred_mean = preds.mean(dim=0)
    epistemic_var = preds.var(dim=0, unbiased=False)
    aleatoric_var = aleatoric_vars.mean(dim=0)
    return pred_mean, aleatoric_var, epistemic_var
# gradient norm utility
def grad_norm(model):
    tot = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            tot += p.grad.data.norm(2).item()**2
            count += 1
    return math.sqrt(tot) if count>0 else 0.0
# ---------- Training / Validation functions ----------
def train_epoch(model, loader, opt, scalers, epoch):
    model.train()
    total_loss = 0.0
    total_data = 0.0
    total_phys = 0.0
    total_grad = 0.0
    n = 0
    wp = get_stage_weights(epoch, CFG)
    for scalars, scalars_raw, target in loader:
        scalars = scalars.to(DEVICE); scalars_raw = scalars_raw.to(DEVICE); target = target.to(DEVICE)
        opt.zero_grad()
        out = model(scalars)
        q_mu = out[:, 0]
        q_logvar = out[:, 1]
        # -------- data loss --------
        data_loss = CFG['w_q'] * gaussian_nll_loss(q_mu, q_logvar, target)
        # ---- physics loss ----
        q_pred_phys = scalers['q'].inverse_torch(q_mu, device=DEVICE)
        q_phys = compute_eurotop_q_torch(scalars_raw)
        phys_loss = wp * F.mse_loss(torch.log(q_pred_phys + EPS), torch.log(q_phys + EPS))
        loss = data_loss + phys_loss
        if not torch.isfinite(loss):
            print("Warning: non-finite loss; skipping step", loss.item())
            continue
        loss.backward()
        gn = grad_norm(model)
        opt.step()
        bs = scalars.shape[0]
        total_loss += loss.item() * bs
        total_data += data_loss.item() * bs
        total_phys += phys_loss.item() * bs
        total_grad += gn * bs
        n += bs
    return total_loss / max(1,n), total_data / max(1,n), total_phys / max(1,n), (total_grad / max(1,n))
def validate(model, val_loader, scalers, epoch):
    model.eval()
    total_loss = 0.0
    total_data = 0.0
    total_phys = 0.0
    n = 0
    preds = {"q": []}
    trues = {"q": []}
    wp = get_stage_weights(epoch, CFG)
    with torch.no_grad():
        for scalars, scalars_raw, target in val_loader:
            scalars = scalars.to(DEVICE); scalars_raw = scalars_raw.to(DEVICE); target = target.to(DEVICE)
            out = model(scalars)
            q_mu = out[:, 0]
            q_logvar = out[:, 1]
            # -------- data loss --------
            data_loss = CFG['w_q'] * gaussian_nll_loss(q_mu, q_logvar, target)
            # ---- physics loss ----
            q_pred_phys = scalers['q'].inverse_torch(q_mu, device=DEVICE)
            q_phys = compute_eurotop_q_torch(scalars_raw)
            phys_loss = wp * F.mse_loss(torch.log(q_pred_phys + EPS), torch.log(q_phys + EPS))
            loss = data_loss + phys_loss
            bs = scalars.shape[0]
            total_loss += loss.item() * bs
            total_data += data_loss.item() * bs
            total_phys += phys_loss.item() * bs
            n += bs
            # predictions for metrics (physical space)
            q_np = scalers['q'].inverse_torch(q_mu, device=DEVICE).cpu().numpy()
            q_t = scalers['q'].inverse(target.cpu().numpy())
            preds["q"].extend(q_np)
            trues["q"].extend(q_t)
    return total_loss / max(1,n), total_data / max(1,n), total_phys / max(1,n), preds, trues
def get_predictions(model, loader, scalers, mc_samples=0):
    preds = []
    trues = []
    with torch.no_grad():
        for scalars, scalars_raw, target in loader:
            scalars = scalars.to(DEVICE); scalars_raw = scalars_raw.to(DEVICE)
            if mc_samples > 0:
                pred_mean, _, _ = mc_predict(model, scalars, scalars_raw, scalers, mc_samples=mc_samples)
                q_np = scalers['q'].inverse_torch(pred_mean, device=DEVICE).cpu().numpy()
            else:
                out = model(scalars)
                q_mu = out[:,0]
                q_np = scalers['q'].inverse_torch(q_mu, device=DEVICE).cpu().numpy()
            q_t = scalers['q'].inverse(target.cpu().numpy())
            preds.extend(q_np)
            trues.extend(q_t)
    return np.array(preds), np.array(trues)
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res / (ss_tot + 1e-8))
def compute_metrics(preds, trues):
    def stats(a, b):
        diff = a - b
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        bias = float(np.mean(diff))
        r2 = r2_score(b, a)  # true, pred
        return {"MAE":mae, "RMSE":rmse, "bias":bias, "R2":r2}
    q_p = np.array(preds["q"]); q_t = np.array(trues["q"])
    q_stats = stats(q_p, q_t) if q_p.size > 0 else {}
    return {"q": q_stats}
def plot_scatter(q_true, q_pred, set_name, file_path):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(q_true, q_pred, alpha=0.5, s=10, color='blue')
    min_val = max(1e-10, min(np.min(q_true[q_true>0]), np.min(q_pred[q_pred>0]))) / 2
    max_val = max(np.max(q_true), np.max(q_pred)) * 2
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Observed $q$ (m$^3$/s/m)')
    ax.set_ylabel(r'Predicted $q$ (m$^3$/s/m)')
    r2_val = r2_score(q_true, q_pred)
    # ax.text(0.05, 0.95, f'$R^2 = {r2_val:.3f}$', transform=ax.transAxes, va='top')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.savefig(file_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
def plot_history(history, key, label, file_path):
    plt.figure(figsize=(4, 3))
    plt.plot(history[key], label=label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.savefig(file_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
def plot_loss_history(history, file_path):
    plt.figure(figsize=(4, 3))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.savefig(file_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
# ---------- MAIN ----------
def main():
    # ---------- EDIT THESE PATHS ----------
    clash_path = r"CLASH database.csv" # Replace with actual path to CLASH .csv file
    # --------------------------------------
    events, q_arr = parse_clash(clash_path)
    N = len(q_arr)
    print("Loaded CLASH sets:", N)
    idx = np.random.permutation(N)
    ntr = max(1, int(0.7 * N)); nval = max(1, int(0.15 * N)); nte = N - ntr - nval
    tr_idx = idx[:ntr]; va_idx = idx[ntr:ntr+nval]; te_idx = idx[ntr+nval:]
    print("Split (train,val,test):", len(tr_idx), len(va_idx), len(te_idx))
    scalers = {}
    scalers['q'] = Scaler(log=True); scalers['q'].fit(q_arr[tr_idx])
    scalar_arr = events.iloc[tr_idx].values.astype(np.float32)
    scal_mean = np.mean(scalar_arr, axis=0).astype(np.float32)
    scal_std = np.std(scalar_arr, axis=0).astype(np.float32) + 1e-9
    scalers['scalars'] = {'mean': scal_mean, 'std': scal_std}
    import pickle
    with open(os.path.join(CKPT_DIR, "scaler_overtopping.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    print("Saved scaler to", os.path.join(CKPT_DIR, "scaler_overtopping.pkl"))
    tr_events = events.iloc[tr_idx].reset_index(drop=True)
    va_events = events.iloc[va_idx].reset_index(drop=True)
    te_events = events.iloc[te_idx].reset_index(drop=True)
    tr_q = q_arr[tr_idx]; va_q = q_arr[va_idx]; te_q = q_arr[te_idx]
    train_ds = OvertoppingSurrogateDataset(tr_events, tr_q, scalers)
    val_ds = OvertoppingSurrogateDataset(va_events, va_q, scalers)
    test_ds = OvertoppingSurrogateDataset(te_events, te_q, scalers)
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False, collate_fn=collate_fn)
    model = SurrogateModel(CFG).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    best_val = 1e12; epochs_no_improve = 0
    history = {"train_loss":[], "train_data_loss":[], "train_phys_loss":[], "train_grad_norm":[], "val_loss":[], "val_data_loss":[], "val_phys_loss":[]}
    for ep in range(1, CFG['epochs']+1):
        t0 = time.time()
        train_loss, train_data_loss, train_phys_loss, train_grad = train_epoch(model, train_loader, opt, scalers, ep)
        val_loss, val_data_loss, val_phys_loss, _, _ = validate(model, val_loader, scalers, ep)
        history['train_loss'].append(train_loss); history['train_data_loss'].append(train_data_loss); history['train_phys_loss'].append(train_phys_loss); history['train_grad_norm'].append(train_grad)
        history['val_loss'].append(val_loss); history['val_data_loss'].append(val_data_loss); history['val_phys_loss'].append(val_phys_loss)
        dt = time.time() - t0
        if ep % CFG['print_every'] == 0:
            print(f"Ep {ep:03d} train_loss={train_loss:.6f} (data={train_data_loss:.6f}, phys={train_phys_loss:.6f}, grad={train_grad:.4f}) val_loss={val_loss:.6f} (data={val_data_loss:.6f}, phys={val_phys_loss:.6f}) time={dt:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_model.pth"))
            epochs_no_improve = 0
            print(" -> saved best @ ep", ep)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CFG['patience']:
                print("Early stopping.")
                break
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "final_model.pth"))
    # save loss plots
    plot_loss_history(history, os.path.join(PLOTS_DIR, "loss_curve.png"))
    plot_history(history, "train_data_loss", "Train Data Loss", os.path.join(PLOTS_DIR, "loss_data_train.png"))
    plot_history(history, "val_data_loss", "Val Data Loss", os.path.join(PLOTS_DIR, "loss_data_val.png"))
    plot_history(history, "train_phys_loss", "Train Phys Loss", os.path.join(PLOTS_DIR, "loss_phys_train.png"))
    plot_history(history, "val_phys_loss", "Val Phys Loss", os.path.join(PLOTS_DIR, "loss_phys_val.png"))
    plot_history(history, "train_grad_norm", "Train Grad Norm", os.path.join(PLOTS_DIR, "grad_norm.png"))
    # --- final test with MC Dropout ---
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_model.pth"), map_location=DEVICE))
    model.to(DEVICE)
    # Get predictions for train and test with MC
    preds_train, trues_train = get_predictions(model, train_loader, scalers, mc_samples=CFG['mc_samples'])
    preds_test, trues_test = get_predictions(model, test_loader, scalers, mc_samples=CFG['mc_samples'])
    # Scatter plots
    plot_scatter(trues_train, preds_train, "Train", os.path.join(PLOTS_DIR, "scatter_train_log.png"))
    plot_scatter(trues_test, preds_test, "Test", os.path.join(PLOTS_DIR, "scatter_test_log.png"))
    # Metrics
    train_metrics = compute_metrics({"q": preds_train}, {"q": trues_train})
    test_metrics = compute_metrics({"q": preds_test}, {"q": trues_test})
    print("TRAIN METRICS:", json.dumps(train_metrics, indent=2, default=float))
    print("TEST METRICS:", json.dumps(test_metrics, indent=2, default=float))
    # save metrics + history
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump({"train_metrics": train_metrics, "test_metrics": test_metrics, "history": history}, f, indent=2, default=float)
    print("Saved outputs to", REPORTS_DIR)
if __name__ == "__main__":
    main()