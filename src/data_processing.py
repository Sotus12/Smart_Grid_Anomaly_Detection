import os
import math
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_CSV = r"C:\SOFTWARE\DL Project\dataset\smart_grid_dataset.csv"


def read_data(csv_path=DEFAULT_CSV):
    df = pd.read_csv(csv_path)
    return df


def infer_feature_columns(df):
    candidates = ['voltage','current','power_consumption','reactive_power','power_factor','solar_power','wind_power','temperature','humidity']
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return cols


def build_labels(df):
    # Create binary anomaly label from known fault indicators
    labels = pd.Series(0, index=df.index)
    possible = ['Overload Condition','Transformer Fault','overload','transformer_fault','fault']
    for p in possible:
        if p in df.columns:
            labels = labels | (df[p].astype(str).str.contains('1|True|true|yes|Yes', na=False))
    return labels.astype(int)


def sliding_windows(arr, window_size=128, step=64):
    n = len(arr)
    if n < window_size:
        return []
    windows = []
    for start in range(0, n - window_size + 1, step):
        windows.append(arr[start:start+window_size])
    return windows


def make_spectrogram(signal_window, nperseg=64, noverlap=32):
    f, t, Sxx = signal.spectrogram(signal_window, nperseg=nperseg, noverlap=noverlap)
    Sxx_log = np.log1p(Sxx)
    return Sxx_log


def save_spectrogram_image(Sxx, out_path, dpi=100):
    plt.figure(figsize=(2.24,2.24), dpi=dpi)
    plt.axis('off')
    plt.imshow(Sxx, aspect='auto', origin='lower', cmap='viridis')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_spectrogram_dataset(csv_path=DEFAULT_CSV, out_dir=r"C:\SOFTWARE\DL Project\outputs\spectrograms",
                                 window_size=128, step=64, signal_col=None, train_frac=0.7, val_frac=0.15, test_frac=0.15,
                                 max_windows_per_class=None, anomaly_threshold=0.1):
    os.makedirs(out_dir, exist_ok=True)
    df = read_data(csv_path)
    labels = build_labels(df)
    features = infer_feature_columns(df)
    if signal_col is None:
        # prefer columns mentioning 'power' or 'consumption' (case-insensitive)
        signal_col = None
        for key in ['power consumption','power_consumption','power','load','predicted load','grid supply']:
            for col in df.columns:
                if key in col.lower():
                    signal_col = col
                    break
            if signal_col:
                break
        if signal_col is None:
            # fallback to first numeric feature
            signal_col = features[0]
    print('Using signal column:', signal_col)

    # group continuous recording into a single series per file; we treat entire CSV as one time series
    series = df[signal_col].fillna(method='ffill').fillna(0).values
    lbl_series = labels.values

    # create windows with majority label in window
    windows = []
    window_labels = []
    idx = 0
    total = len(series)
    # label window as anomaly if fraction of fault indicators in the window >= anomaly_threshold
    anomaly_threshold = 0.1
    while idx + window_size <= total:
        win = series[idx:idx+window_size]
        frac = float(lbl_series[idx:idx+window_size].sum())/float(window_size)
        lab = 1 if frac >= anomaly_threshold else 0
        windows.append((win, lab))
        idx += step
    print(f'Generated {len(windows)} windows')

    # split by simple stratified split
    X = [w for w,l in windows]
    y = [l for w,l in windows]
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=(1-train_frac), random_state=42)
    rel = val_frac/(val_frac+test_frac)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=(1-rel), random_state=42)

    splits = [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]
    for split_name, Xs, ys in splits:
        for cls in [0,1]:
            d = os.path.join(out_dir, split_name, 'normal' if cls==0 else 'anomaly')
            os.makedirs(d, exist_ok=True)
        counts = {0:0,1:0}
        for i,(w,l) in enumerate(zip(Xs, ys)):
            if max_windows_per_class and counts[l] >= max_windows_per_class:
                continue
            S = make_spectrogram(w)
            fname = f"{split_name}_{i}_lbl{l}.png"
            out_path = os.path.join(out_dir, split_name, 'normal' if l==0 else 'anomaly', fname)
            save_spectrogram_image(S, out_path)
            counts[l] += 1
        print(f'Wrote split {split_name} counts:', counts)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=DEFAULT_CSV)
    parser.add_argument('--out', default=r"C:\SOFTWARE\DL Project\outputs\spectrograms")
    parser.add_argument('--window', type=int, default=128)
    parser.add_argument('--step', type=int, default=64)
    parser.add_argument('--signal')
    parser.add_argument('--max-per-class', type=int, default=2000)
    parser.add_argument('--anomaly-threshold', type=float, default=0.1)
    args = parser.parse_args()
    generate_spectrogram_dataset(csv_path=args.csv, out_dir=args.out, window_size=args.window,
                                 step=args.step, signal_col=args.signal, max_windows_per_class=args.max_per_class,
                                 anomaly_threshold=args.anomaly_threshold)
