import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from BNN_KWS_GSC import BNN_KWS, waveform_to_logmel

ckpt = torch.load("bnn_KWS.pt", map_location="cpu")
args = ckpt["args"]
model = BNN_KWS(
    num_classes=ckpt["num_classes"],
    bin_first=args["bin_first"],
    bin_last=args["bin_last"],
    use_scale=not args["no_scale"],
)
model.load_state_dict(ckpt["model"])
model.eval()

# 确保 Speech Commands 数据集已下载到当前目录
SPEECHCOMMANDS(root="./speech_commands", download=True)

waveform, sr = torchaudio.load("D:/Vitis/USERS/10_Zedboard_audio_in/KWS/yes.wav")
feat = waveform_to_logmel(waveform, sr).unsqueeze(0)  # [1, 1, 40, 98]
with torch.no_grad():
    logits = model(feat)
    pred_idx = logits.argmax(dim=1).item()

print(pred_idx)
# # 可选：恢复标签字符串
# from torchaudio.datasets import SPEECHCOMMANDS
# labels = sorted({sample[2] for sample in SPEECHCOMMANDS("./speech_commands", subset="training", download=True)})
# pred_label = labels[pred_idx]
# print(pred_label)
