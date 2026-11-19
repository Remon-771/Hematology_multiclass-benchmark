"""Deep tabular models: FT-Transformer, residual MLP and (optional) TabNet (notebook §10)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import SEED
from .preprocessing import make_preprocessor

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:  # pragma: no cover
    TORCH_OK = False
    device = "cpu"

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_OK = TORCH_OK
except ImportError:  # pragma: no cover
    TABNET_OK = False


if TORCH_OK:
    class ResBlock(nn.Module):
        def __init__(self, dim, hidden, dropout=0.2):
            super().__init__()
            self.bn = nn.BatchNorm1d(dim); self.fc1 = nn.Linear(dim, hidden); self.fc2 = nn.Linear(hidden, dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.bn(x); h = F.relu(self.fc1(h)); h = self.drop(h); h = self.fc2(h)
            return x + h

    class TabularResNet(nn.Module):
        def __init__(self, in_dim, n_classes, hidden=128, n_blocks=4, dropout=0.2):
            super().__init__()
            self.inp = nn.Linear(in_dim, hidden)
            self.blocks = nn.ModuleList([ResBlock(hidden, hidden * 2, dropout) for _ in range(n_blocks)])
            self.out_bn = nn.BatchNorm1d(hidden); self.out = nn.Linear(hidden, n_classes)

        def forward(self, x):
            x = self.inp(x)
            for b in self.blocks:
                x = b(x)
            return self.out(self.out_bn(x))

    class FTTransformer(nn.Module):
        def __init__(self, n_features, n_classes, d_token=64, n_layers=3, n_heads=8, dropout=0.15):
            super().__init__()
            self.feature_embed = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
            self.feature_bias = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
                                                   dropout=dropout, activation="gelu", batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_token); self.head = nn.Linear(d_token, n_classes)

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.feature_embed.unsqueeze(0) + self.feature_bias.unsqueeze(0)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            enc = self.encoder(torch.cat([cls, tok], dim=1))
            return self.head(self.norm(enc[:, 0, :]))

    def to_loader(Xa, ya, batch_size=256, shuffle=True):
        ds = TensorDataset(torch.tensor(Xa, dtype=torch.float32), torch.tensor(ya, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def train_torch_model(model, train_loader, val_loader, n_epochs, lr=1e-3, patience=12, name="model"):
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        crit = nn.CrossEntropyLoss()
        best_val_loss, best_state, no_improve = float("inf"), None, 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        for epoch in range(n_epochs):
            model.train(); train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); out = model(xb); loss = crit(out, yb); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval(); val_loss, correct = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb); loss = crit(out, yb)
                    val_loss += loss.item() * xb.size(0)
                    correct += (out.argmax(1) == yb).sum().item()
            val_loss /= len(val_loader.dataset); val_acc = correct / len(val_loader.dataset)
            sched.step(val_loss)
            history["train_loss"].append(train_loss); history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss, best_state, no_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    def torch_probs(model, X_proc):
        model.eval()
        with torch.no_grad():
            return F.softmax(model(torch.tensor(X_proc, dtype=torch.float32).to(device)), dim=1).cpu().numpy()


def prepare_dl_data(split, seed: int = SEED):
    """Preprocessing fitted once on the training sub-split only; SMOTE on the sub-split only."""
    prep = make_preprocessor(split.feature_names)
    X_tr_proc = prep.fit_transform(split.X_tr)
    X_val_proc = prep.transform(split.X_val)
    X_tr_bal, y_tr_bal = SMOTE(random_state=seed, k_neighbors=5).fit_resample(X_tr_proc, split.y_tr)
    return prep, X_tr_bal, y_tr_bal, X_val_proc


def train_all_dl(split, cfg):
    """Train ResNet-MLP, FT-Transformer and (if installed) TabNet on the training sub-split."""
    if not TORCH_OK:
        return {}, {}, None
    prep, X_tr_bal, y_tr_bal, X_val_proc = prepare_dl_data(split)
    n_features = X_tr_bal.shape[1]
    train_loader = to_loader(X_tr_bal, y_tr_bal, shuffle=True)
    val_loader = to_loader(X_val_proc, split.y_val, shuffle=False)
    dl_results, dl_histories = {}, {}
    m, h = train_torch_model(TabularResNet(n_features, split.n_classes), train_loader, val_loader, cfg.dl_epochs, name="Tabular ResNet-MLP")
    dl_results["Tabular ResNet-MLP"], dl_histories["Tabular ResNet-MLP"] = m, h
    m, h = train_torch_model(FTTransformer(n_features, split.n_classes), train_loader, val_loader, cfg.dl_epochs, name="FT-Transformer")
    dl_results["FT-Transformer"], dl_histories["FT-Transformer"] = m, h
    if TABNET_OK:
        tabnet = TabNetClassifier(n_d=32, n_a=32, n_steps=5, gamma=1.5, seed=SEED, verbose=0, optimizer_params=dict(lr=2e-2))
        tabnet.fit(X_tr_bal, y_tr_bal, eval_set=[(X_val_proc, split.y_val)], eval_metric=["logloss"],
                   max_epochs=cfg.tabnet_epochs, patience=15, batch_size=256, virtual_batch_size=64)
        dl_results["TabNet"] = tabnet
        dl_histories["TabNet"] = {"train_loss": tabnet.history["loss"], "val_loss": tabnet.history["val_0_logloss"]}
    else:
        print("pytorch-tabnet not installed; run `pip install pytorch-tabnet` to include TabNet.")
    return dl_results, dl_histories, prep


def dl_validation_table(dl_results, prep, split) -> pd.DataFrame:
    X_val_proc = prep.transform(split.X_val)
    rows = []
    for name, model in dl_results.items():
        probs = model.predict_proba(X_val_proc) if name == "TabNet" else torch_probs(model, X_val_proc)
        preds = probs.argmax(1)
        rows.append({"Model": name, "Val_Accuracy": accuracy_score(split.y_val, preds),
                     "Val_MacroF1": f1_score(split.y_val, preds, average="macro"),
                     "Val_WeightedF1": f1_score(split.y_val, preds, average="weighted"),
                     "Val_ROC_AUC": roc_auc_score(split.y_val, probs, average="macro", multi_class="ovr")})
    return pd.DataFrame(rows).sort_values("Val_MacroF1", ascending=False) if rows else pd.DataFrame()
