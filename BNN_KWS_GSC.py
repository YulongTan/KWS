import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader

# -----------------------------
# 复用：随机种子
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

# -----------------------------
# 与示例一致的数据预处理
# -----------------------------
DATASET_PATH = "./speech_commands"

TARGET_SR = 16000
N_MELS = 40
N_FFT = 400
HOP = 160
WIN = 400

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    win_length=WIN,
    hop_length=HOP,
    n_mels=N_MELS
)

def waveform_to_logmel(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    # 到单声道 [1, T]
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # 重采样到16k
    if sample_rate != TARGET_SR:
        resample = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SR)
        waveform = resample(waveform)

    # 固定到1秒（16000点）
    TGT_LEN = TARGET_SR
    cur_len = waveform.shape[-1]
    if cur_len < TGT_LEN:
        pad_len = TGT_LEN - cur_len
        waveform = F.pad(waveform, (0, pad_len))
    elif cur_len > TGT_LEN:
        start = (cur_len - TGT_LEN) // 2
        waveform = waveform[:, start:start + TGT_LEN]

    # Mel & log  -> [1, 40, 98] (后续当作 [B,1,40,98])
    mel = mel_transform(waveform).clamp(min=1e-6).log()
    return mel

def make_dataloaders(batch_size: int, data_root: str):
    dataset_path = data_root if data_root else DATASET_PATH

    train_set = torchaudio.datasets.SPEECHCOMMANDS(
        root=dataset_path, subset="training", download=True)
    val_set = torchaudio.datasets.SPEECHCOMMANDS(
        root=dataset_path, subset="validation", download=True)
    test_set = torchaudio.datasets.SPEECHCOMMANDS(
        root=dataset_path, subset="testing", download=True)

    # 统一标签映射（以训练集为准）
    labels = sorted(list(set(dat[2] for dat in train_set)))
    label2idx = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)

    class _Wrap(torch.utils.data.Dataset):
        def __init__(self, base, l2i):
            self.base = base
            self.l2i = l2i
        def __len__(self): return len(self.base)
        def __getitem__(self, idx):
            waveform, sr, label, *_ = self.base[idx]
            x = waveform_to_logmel(waveform, sr)     # [1,40,98]
            y = self.l2i[label]
            return x, y

    train_ds = _Wrap(train_set, label2idx)
    val_ds   = _Wrap(val_set,   label2idx)
    test_ds  = _Wrap(test_set,  label2idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes

# -----------------------------
# BNN 组件：二值化+STE
# -----------------------------
class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 二值激活：+1/-1
        return x.sign().clamp(min=-1., max=1.)
    @staticmethod
    def backward(ctx, grad_output):
        # 直通估计器（identity）
        return grad_output

def binarize(x):
    return SignSTE.apply(x)

def weight_binarize(w: torch.Tensor, use_scale: bool = True):
    # 权值二值化，带简单缩放（E[|w|]）
    if use_scale:
        alpha = w.abs().mean(dim=list(range(1, w.dim())), keepdim=True)
        return binarize(w), alpha
    else:
        return binarize(w), None

class BinaryActivation(nn.Module):
    def forward(self, x):
        return binarize(x)

class BinaryConv2d(nn.Module):
    """
    二值卷积：训练存实权w，前向用二值化权重（可选缩放因子）。
    常见做法：先BN再二值激活，再Conv（这里用 Conv->BN->BinAct 的简单序列，效果也可）。
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False, use_scale=True):
        super().__init__()
        self.real_weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        nn.init.kaiming_normal_(self.real_weight, nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.bn = nn.BatchNorm2d(out_ch)
        self.use_scale = use_scale
        self.act = BinaryActivation()

        self.stride = s
        self.padding = p

    def forward(self, x):
        bw, alpha = weight_binarize(self.real_weight, self.use_scale)
        w = bw if alpha is None else bw * alpha  # XNOR-Net风格缩放
        x = F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding)
        x = self.bn(x)
        x = self.act(x)
        return x

class BinaryLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False, use_scale=True):
        super().__init__()
        self.real_weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_normal_(self.real_weight, nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.bn = nn.BatchNorm1d(out_f)
        self.use_scale = use_scale
        self.act = BinaryActivation()

    def forward(self, x):
        bw, alpha = weight_binarize(self.real_weight, self.use_scale)
        w = bw if alpha is None else bw * alpha
        x = F.linear(x, w, self.bias)
        x = self.bn(x)
        x = self.act(x)
        return x

# -----------------------------
# BNN 模型（KWS）
# 输入：[B,1,40,98]
# 默认首层/末层保留FP（--bin-first/--bin-last 可切换）
# -----------------------------
class BNN_KWS(nn.Module):
    def __init__(self, num_classes: int, bin_first: bool = False, bin_last: bool = False, use_scale: bool = True):
        super().__init__()
        self.bin_first = bin_first
        self.bin_last  = bin_last
        self.use_scale = use_scale

        # stem：40x98 -> 20x49
        if bin_first:
            self.conv1 = BinaryConv2d(1, 32, k=3, s=1, p=1, use_scale=use_scale)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        self.pool1 = nn.MaxPool2d(2)

        # 20x49 -> 10x24
        self.bin2 = BinaryConv2d(32, 64, k=3, s=1, p=1, use_scale=use_scale)
        self.pool2 = nn.MaxPool2d(2)

        # 10x24 -> 5x12
        self.bin3 = BinaryConv2d(64, 128, k=3, s=1, p=1, use_scale=use_scale)
        self.pool3 = nn.MaxPool2d(2)

        # 展平
        self.gap = nn.AdaptiveAvgPool2d((5, 5))  # -> [128,5,5]
        feat_dim = 128 * 5 * 5  # 3200

        # 分类头
        if bin_last:
            self.fc1 = BinaryLinear(feat_dim, 256, use_scale=use_scale)
            self.fc_out = BinaryLinear(256, num_classes, use_scale=use_scale)
            # 最后一层二值化后再线性->BN->BinAct，输出用于CE，等价于对数iterm不太稳定
            # 更稳定做法：最后一层保持FP，这里仅在 bin_last=True 时才二值
        else:
            self.fc1 = BinaryLinear(feat_dim, 256, use_scale=use_scale)
            self.fc_out = nn.Linear(256, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)    # [B,32,40,98] or bin
        x = self.pool1(x)    # [B,32,20,49]

        x = self.bin2(x)     # [B,64,20,49]
        x = self.pool2(x)    # [B,64,10,24]

        x = self.bin3(x)     # [B,128,10,24]
        x = self.pool3(x)    # [B,128,5,12]

        x = self.gap(x)      # [B,128,5,5]
        x = torch.flatten(x, 1)  # [B,3200]

        x = self.fc1(x)      # [B,256] (binary)
        x = self.fc_out(x)   # [B,C]   (fp or binary)
        return x

# -----------------------------
# 评估
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, n = 0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n += yb.numel()
    return 100.0 * correct / max(1, n)

# -----------------------------
# 训练
# -----------------------------
def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, val_loader, test_loader, num_classes = make_dataloaders(args.batch_size, args.data_root)

    model = BNN_KWS(num_classes=num_classes,
                    bin_first=args.bin_first,
                    bin_last=args.bin_last,
                    use_scale=not args.no_scale).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # BNN梯度可能较噪，做个范数裁剪更稳
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running += loss.item()

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:03d}: LR={optimizer.param_groups[0]['lr']:.3e} "
              f"TrainLoss={running/len(train_loader):.4f}  "
              f"ValAcc={val_acc:.2f}%  TestAcc={test_acc:.2f}%")

        if val_acc > best_val and args.ckpt:
            best_val = val_acc
            os.makedirs(os.path.dirname(args.ckpt), exist_ok=True) if os.path.dirname(args.ckpt) else None
            torch.save({"model": model.state_dict(),
                        "num_classes": num_classes,
                        "args": vars(args)}, args.ckpt)
            print(f"[Info] Saved checkpoint to {args.ckpt} (ValAcc={val_acc:.2f}%)")

# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BNN for KWS (SPEECHCOMMANDS, Log-Mel, BP training)")
    p.add_argument("--data-root", type=str, default="./speech_commands")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    # BNN 细节
    p.add_argument("--bin-first", action="store_true", help="binarize the first conv layer (default: FP)")
    p.add_argument("--bin-last", action="store_true", help="binarize the last classifier layer (default: FP)")
    p.add_argument("--no-scale", action="store_true", help="disable XNOR scaling factor (default: enabled)")
    p.add_argument("--ckpt", type=str, default="bnn_KWS.pt", help="path to save best checkpoint by ValAcc")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
