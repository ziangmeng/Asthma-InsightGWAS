# infer_transformer_asthma.py
# infer_transformer_asthma.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import random
import pickle

# -------------------- Paths (relative to this script) --------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INFER_PATH = BASE_DIR / "data" / "Inference_Data_Asthma_10pct.txt"
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "asthma_transformer.pth"
DEFAULT_SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"        
DEFAULT_OUT_PATH   = BASE_DIR / "output" / "asthma_predictions_10pct.txt"

FEATURE_COLS = [
    'chr_hg38', 'bpos_GCST010042', 'ld', 'beta_GCST010042',
    'se_GCST010042', 'pval_GCST010042', 'n_GCST010042',
    'GTEx_V8_cis_eqtl', 'not_GTEx_eqtl', 'mqtl_EAS', 'mqtl_EUR',
    'sqtl', 'scqtl', 'consensus_footprints_and_motifs_hg38',
    'encode', 'report'
]

# -------------------- Model --------------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, d_model=20, output_dim=1):
        super().__init__()
        self.input_mapping = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_mapping(x)   # (batch, d_model)
        x = x.unsqueeze(1)          # (batch, 1, d_model)
        x = x.permute(1, 0, 2)      # (1, batch, d_model)
        x = self.transformer(x)     # (1, batch, d_model)
        x = x[0, :, :]              # (batch, d_model)
        x = self.fc(x)              # (batch, 1)
        x = self.sigmoid(x)
        return x

# -------------------- Utils --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_and_prepare_infer_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    if 'ld' not in df.columns and 'ldscore' in df.columns:
        df = df.rename(columns={'ldscore': 'ld'})

    if 'snp' not in df.columns:
        df['snp'] = np.arange(len(df)).astype(str)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    out = df[['snp'] + FEATURE_COLS].copy()

    for col in FEATURE_COLS:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)

    return out

def try_load_scaler(scaler_path: Path):
    if scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            print(f"[Info] Loaded scaler from: {scaler_path}")
            return scaler
        except Exception as e:
            print(f"[Warn] Failed to load scaler at {scaler_path}: {e}")
    else:
        print(f"[Info] Scaler not found at {scaler_path}, continue without scaling.")
    return None  
# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Inference with saved asthma Transformer model (no retraining)")
    parser.add_argument('--infer_path', type=str, default=str(DEFAULT_INFER_PATH),
                        help='Path to inference input (.txt, tab-separated), default ./data/Inference_Data_Asthma_10pct.txt')
    parser.add_argument('--model_path', type=str, default=str(DEFAULT_MODEL_PATH),
                        help='Path to saved model (.pth), default ./model/asthma_transformer.pth')
    parser.add_argument('--scaler_path', type=str, default=str(DEFAULT_SCALER_PATH),
                        help='Optional path to saved scaler (.pkl). If missing, no scaling is applied.')
    parser.add_argument('--output_path', type=str, default=str(DEFAULT_OUT_PATH),
                        help='Path to save predictions, default ./output/asthma_predictions_10pct.txt')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    infer_path = Path(args.infer_path)
    model_path = Path(args.model_path)
    scaler_path = Path(args.scaler_path)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    infer_df = load_and_prepare_infer_df(infer_path)
    snps = infer_df['snp'].astype(str).values
    X_infer = infer_df[FEATURE_COLS].values


    scaler = try_load_scaler(scaler_path)
    if scaler is not None:
        try:
            X_infer = scaler.transform(X_infer)
        except Exception as e:
            print(f"[Warn] Scaler.transform failed: {e}\nProceeding without scaling.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_infer.shape[1]
    model = TransformerClassifier(
        input_dim=input_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        d_model=args.d_model
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()


    tensor = torch.tensor(X_infer, dtype=torch.float32)
    ds = TensorDataset(tensor)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    probs_list = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            out = model(xb)  
            probs_list.append(out.squeeze(1).cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    preds = (probs > args.threshold).astype(int)

    out_df = pd.DataFrame({'snp': snps, 'prob': probs, 'pred': preds})
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved predictions to: {out_path}")

if __name__ == "__main__":
    main()
