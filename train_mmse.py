import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import copy
from thop import profile
from torchinfo import summary
import time

from client_model import ClientModel
from server_model import ServerModel
from client1 import Client    
from server1 import MainServer  
from robust_aggregation import fedserver
from channel_simulation import CommunicationChannel, PixelNoiseInjector

EXPERIMENT_NAME = "Hybrid_SNR_EMA_Denoising"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{EXPERIMENT_NAME}] Using device: {device}")

K = 5
rounds = 50
local_epochs = 3  
batch_size = 128
lr = 0.002

ENABLE_CHANNEL_NOISE = True
ENABLE_PIXEL_NOISE = False

AGGREGATION_METHOD = 'fedavg'

SNR_DB = 5
CHANNEL_GAIN = 1.0
BIT_ERROR_RATE = 0.001
PIXEL_NOISE_STD = 0.05

# 實驗控制面板
SIMULATE_IDEAL_CHANNEL = False  # 是否模擬「無雜訊」的天花板環境
ENABLE_FORWARD_SCALING = False   # 是否開啟前向特徵 SNR 縮放
ENABLE_BACKWARD_EMA    = False   # 是否開啟反向梯度 SNR EMA

print(f"\n=== [{EXPERIMENT_NAME}] Settings ===")
print(f"Channel noise: On (Base SNR={SNR_DB}dB)")
print(f"Denoising 1 (Forward) : Dynamic SNR Weight Scaling on A_k")
print(f"Denoising 2 (Backward): SNR-Driven EMA on Gradients dA_k")
print(f"Aggregation method: {AGGREGATION_METHOD}")

print("\nPreparing DeepGlobe dataset...")
from deepglobe import DeepGlobeDataset

full_dataset = DeepGlobeDataset(root_dir='C:/Users/winlab/Desktop/Flora/deepglobe', split=None, img_size=(64, 64))
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset_full, test_dataset = random_split(full_dataset, [train_size, test_size])

nk_list = [len(train_dataset_full) // K] * K
nk_list[-1] += len(train_dataset_full) - sum(nk_list)
client_subsets = random_split(train_dataset_full, nk_list)

dataloaders = [DataLoader(sub, batch_size=batch_size, shuffle=True) for sub in client_subsets]
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

channels = []
pixel_injectors = []
for i in range(K):
    channel = CommunicationChannel(snr_db=SNR_DB, channel_gain=CHANNEL_GAIN, bit_error_rate=BIT_ERROR_RATE)
    pixel_injector = PixelNoiseInjector(noise_std=PIXEL_NOISE_STD) if ENABLE_PIXEL_NOISE else None
    channels.append(channel)
    pixel_injectors.append(pixel_injector)

global_server_model = ServerModel(num_classes=7).to(device)
server_lr = 0.001
main_server = MainServer(global_server_model, device, lr=server_lr, denoiser=None)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(main_server.optimizer, T_max=rounds, eta_min=1e-5)

clients = []
for i in range(K):
    c_model = ClientModel().to(device)
    c_instance = Client(model=c_model, device=device, lr=lr, channel=channels[i], pixel_noise_injector=pixel_injectors[i])
    clients.append(c_instance)

print("\nSynchronizing initial client models (t=0) ...")
initial_global_weights = clients[0].model.state_dict()
for c in clients:
    c.model.load_state_dict(initial_global_weights)

def evaluate(c_model, s_model, loader):
    c_model.eval()
    s_model.eval()
    total_pixels = 0
    correct_pixels = 0
    intersection = torch.zeros(7, device=device)
    union = torch.zeros(7, device=device)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            activations = c_model(x)
            outputs = s_model(activations)
            _, predicted = torch.max(outputs, 1)
            
            valid_mask = y != 6
            total_pixels += valid_mask.sum().item()
            correct_pixels += ((predicted == y) & valid_mask).sum().item()

            for cls in range(6):
                pred_cls = predicted == cls
                true_cls = y == cls
                intersection[cls] += (pred_cls & true_cls & valid_mask).sum()
                union[cls] += ((pred_cls | true_cls) & valid_mask).sum()
                
    iou = intersection[:6] / (union[:6] + 1e-6)
    miou = iou.mean().item()
    pixel_acc = correct_pixels / max(total_pixels, 1)
    return pixel_acc, miou

print("\n=== Hardware & Performance Analysis Report ===")

dummy_client_input = torch.randn(1, 3, 64, 64).to(device)
with torch.no_grad():
    dummy_server_input = clients[0].model(dummy_client_input)

print("\n[1] Client Model Parameters & Memory:")
summary(clients[0].model, input_data=(dummy_client_input,), device=device)

print("\n[2] Server Model Parameters & Memory:")

dummy_client_model = copy.deepcopy(clients[0].model)
dummy_server_model = copy.deepcopy(main_server.model)

macs_c, params_c = profile(dummy_client_model, inputs=(dummy_client_input, ), verbose=False)
macs_s, params_s = profile(dummy_server_model, inputs=(dummy_server_input, ), verbose=False)

flops_c = macs_c * 2
flops_s = macs_s * 2

print("\n[3] Computational Complexity (per image):")
print(f" - Client Model FLOPs : {flops_c / 1e6:.2f} Mega-FLOPs (MFLOPs)")
print(f" - Server Model FLOPs : {flops_s / 1e6:.2f} Mega-FLOPs (MFLOPs)")
print(f" - Total System FLOPs : {(flops_c + flops_s) / 1e6:.2f} MFLOPs")

del dummy_client_model
del dummy_server_model

print("\n[4] Latency Measurement (Running 100 warmups & 100 tests)...")
def measure_latency(model, dummy_input, device, num_tests=100):
    model.eval()
    for _ in range(100):
        _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_tests):
        _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    return ((end_time - start_time) / num_tests) * 1000

with torch.no_grad():
    client_latency = measure_latency(clients[0].model, dummy_client_input, device)
    server_latency = measure_latency(main_server.model, dummy_server_input, device)

print(f" - Client Forward Latency : {client_latency:.3f} ms / image")
print(f" - Server Forward Latency : {server_latency:.3f} ms / image")
print(f" - Total Inference Latency: {(client_latency + server_latency):.3f} ms / image\n")

print(f"\nStart training [{EXPERIMENT_NAME}]: Rounds={rounds}, Local Epochs={local_epochs}")
train_losses, test_accuracies, test_mious = [], [], []

# 建立反向梯度的 EMA 緩衝區
client_grad_ema = {i: None for i in range(K)}

for r in range(rounds):
    # 每回合動態更新通道 SNR
    # 每回合動態更新通道 SNR
    round_snrs = []
    for i in range(K):
        # 【修改點 A】：判斷是否為無雜訊的理想環境
        if SIMULATE_IDEAL_CHANNEL:
            new_snr = 100.0  # 100dB 視同完全無雜訊
        else:
            new_snr = SNR_DB + np.random.randn() * 2
            
        clients[i].channel.snr_db = new_snr
        round_snrs.append(new_snr)
    avg_snr = np.mean(round_snrs)

    print(f'\n--- Round {r+1}/{rounds} (Avg SNR={avg_snr:.2f}dB) ---')
    round_loss = 0.0
    total_steps = 0

    for i in range(K):
        for epoch in range(local_epochs):
            for batch_idx, (x, y) in enumerate(dataloaders[i]):
                x, y = x.to(device), y.to(device)

                A_k_received = clients[i].ClientUpdate(x, add_pixel_noise=ENABLE_PIXEL_NOISE)
                A_k_received = torch.nan_to_num(A_k_received, nan=0.0)
                A_k_received = torch.clamp(A_k_received, min=-10.0, max=10.0)

                # ==========================================
                # 機制 1：前向傳播的動態 SNR 特徵縮放
                # ==========================================
                current_snr = clients[i].channel.snr_db
                snr_linear = 10 ** (current_snr / 10.0)
                base_snr_weight = snr_linear / (snr_linear + 1.0)
                
                # 【修改點 B】：如果前向防禦關閉，權重強制設為 1.0 (不縮放)
                actual_forward_weight = base_snr_weight if ENABLE_FORWARD_SCALING else 1.0
                
                A_k_denoised = A_k_received * actual_forward_weight
                A_k_server_input = A_k_denoised.to(device).detach().clone().requires_grad_(True)

                dA_k, loss_val = main_server.ServerUpdate(A_k_server_input, y, clear_grad=True)
                dA_k = torch.clamp(dA_k, min=-0.5, max=0.5)

                # ==========================================
                # 機制 2：反向傳播的 SNR 驅動梯度 EMA
                # ==========================================
                # 【修改點 C】：如果反向防禦關閉，alpha 強制設為 0.0 (只看當下梯度，不看歷史)
                actual_ema_alpha = (1.0 - base_snr_weight) if ENABLE_BACKWARD_EMA else 0.0
                current_b = dA_k.shape[0] 
                
                if client_grad_ema[i] is None or client_grad_ema[i].shape[0] != current_b:
                    client_grad_ema[i] = dA_k.clone()
                else:
                    client_grad_ema[i] = actual_ema_alpha * client_grad_ema[i] + (1.0 - actual_ema_alpha) * dA_k
                
                dA_k_smoothed = client_grad_ema[i].clone()

                clients[i].ClientBackprop(dA_k_smoothed)
                torch.nn.utils.clip_grad_norm_(clients[i].model.parameters(), max_norm=5.0)

                torch.nn.utils.clip_grad_norm_(main_server.model.parameters(), max_norm=5.0)
                main_server.step()

                if i == 0 and batch_idx == 0:
                    grad_norm = dA_k_smoothed.abs().mean().item()
                    first_layer_grad = 0
                    for param in clients[i].model.parameters():
                        if param.grad is not None:
                            first_layer_grad = param.grad.abs().mean().item()
                            break
                    print(f"Client 0 SNR={current_snr:.2f}dB | 前向縮放 W={actual_forward_weight:.4f} | 反向EMA α={actual_ema_alpha:.4f}")
                    print(f"Server回傳平滑梯度={grad_norm:.6e}, Client權重梯度={first_layer_grad:.6f}")

                round_loss += loss_val
                total_steps += 1

    avg_loss = round_loss / total_steps if total_steps > 0 else 0
    train_losses.append(avg_loss)

    client_models_list = [c.model for c in clients]
    aggregated_model = fedserver(client_models_list, nk_list, sum(nk_list), method=AGGREGATION_METHOD, trim_ratio=0.2, f=1)
    global_weights = aggregated_model.state_dict()
    for c in clients:
        c.model.load_state_dict(global_weights)

    test_acc, miou = evaluate(clients[0].model, main_server.model, test_loader)
    test_accuracies.append(test_acc)
    test_mious.append(miou)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Pixel Acc: {test_acc*100:.2f}% | mIoU: {miou:.4f}")

print(f'\n=== [{EXPERIMENT_NAME}] Final Result ===')
print(f'Final Pixel Accuracy: {test_accuracies[-1] * 100:.2f}%')
print(f'Best Pixel Accuracy:  {max(test_accuracies) * 100:.2f}%')
print(f'Final mIoU: {test_mious[-1]:.4f}')
print(f'Best mIoU:  {max(test_mious):.4f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(range(1, rounds + 1), train_losses, marker='o', color='b', markersize=4, label='Train Loss')
axes[0].set_title('Training Loss over Rounds')
axes[0].set_xlabel('Communication Rounds')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

acc_percent = [acc * 100 for acc in test_accuracies]
axes[1].plot(range(1, rounds + 1), acc_percent, marker='s', color='g', markersize=4, label='Pixel Accuracy')
axes[1].set_title('Test Pixel Accuracy over Rounds')
axes[1].set_xlabel('Communication Rounds')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

axes[2].plot(range(1, rounds + 1), test_mious, marker='^', color='r', markersize=4, label='mIoU')
axes[2].set_title('Test mIoU over Rounds')
axes[2].set_xlabel('Communication Rounds')
axes[2].set_ylabel('mIoU')
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].legend()
plt.show()

# ---------------- 視覺化函數 ----------------

class ActivationDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

def visualize_activation_denoising(client, server, dataloader, device, decoder_steps=300):
    mean_t = torch.tensor([0.344, 0.380, 0.407], device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor([0.203, 0.136, 0.114], device=device).view(1, 3, 1, 1)
    mean_np = np.array([0.344, 0.380, 0.407])
    std_np  = np.array([0.203, 0.136, 0.114])

    def to_display(tensor_img):
        img = tensor_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img = img * std_np + mean_np
        return np.clip(img, 0, 1)

    def psnr(recon, target):
        mse = np.mean((recon.astype(np.float32) - target.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))

    client.model.eval()
    server.model.eval()

    x_batch, _ = next(iter(dataloader))
    x_batch = x_batch.to(device)
    x_single = x_batch[0:1]

    print(f"\n[Decoder Training] ({decoder_steps} steps)...")
    decoder = ActivationDecoder().to(device)
    optimizer_d = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    criterion_d = nn.MSELoss()

    for step in range(decoder_steps):
        decoder.train()
        client.model.eval()
        with torch.no_grad():
            act = client.model(x_batch)
        recon = decoder(act)
        loss_d = criterion_d(recon, x_batch)
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

    decoder.eval()

    with torch.no_grad():
        clean_act    = client.model(x_single)
        noisy_act    = client.channel.transmit(clean_act)

        current_snr = client.channel.snr_db
        snr_linear = 10 ** (current_snr / 10.0)
        snr_weight = snr_linear / (snr_linear + 1.0)
        denoised_act = noisy_act * snr_weight

    with torch.no_grad():
        recon_clean    = decoder(clean_act)
        recon_noisy    = decoder(noisy_act)
        recon_denoised = decoder(denoised_act)

    orig_vis      = to_display(x_single)
    vis_clean     = to_display(recon_clean)
    vis_noisy     = to_display(recon_noisy)
    vis_denoised  = to_display(recon_denoised)

    psnr_clean    = psnr(vis_clean,    orig_vis)
    psnr_noisy    = psnr(vis_noisy,    orig_vis)
    psnr_denoised = psnr(vis_denoised, orig_vis)

    delta_psnr = psnr_denoised - psnr_noisy

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Activation Reconstruction Comparison (SNR={client.channel.snr_db:.1f}dB)', fontsize=13)

    panels = [
        (orig_vis,     'Original Input',                           'black'),
        (vis_clean,    f'Reconstructed: Clean\nPSNR={psnr_clean:.1f}dB',    'green'),
        (vis_noisy,    f'Reconstructed: Noisy\nPSNR={psnr_noisy:.1f}dB',    'red'),
        (vis_denoised, f'Reconstructed: SNR Scaled\nPSNR={psnr_denoised:.1f}dB  (Δ{delta_psnr:+.1f}dB)', 'royalblue'),
    ]

    for ax, (img, title, color) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    plt.tight_layout()
    plt.show()

def visualize_feature_maps(client, dataloader, device, num_channels=8):
    client.model.eval()

    x_batch, _ = next(iter(dataloader))
    x_single = x_batch[0:1].to(device)

    with torch.no_grad():
        clean_act = client.model(x_single)
        noisy_act = client.channel.transmit(clean_act.clone())

        current_snr = client.channel.snr_db
        snr_linear = 10 ** (current_snr / 10.0)
        snr_weight = snr_linear / (snr_linear + 1.0)
        denoised_act = noisy_act.clone() * snr_weight

    def act_np(t):
        return t.squeeze(0).cpu().numpy()

    clean_np    = act_np(clean_act)[:num_channels]
    noisy_np    = act_np(noisy_act)[:num_channels]
    denoised_np = act_np(denoised_act)[:num_channels]

    total_C = clean_np.shape[0]
    labels  = ['Clean', 'Noisy', 'SNR Scaled']
    arrays  = [clean_np, noisy_np, denoised_np]
    colors  = ['green', 'red', 'royalblue']

    fig1, axes1 = plt.subplots(3, total_C, figsize=(2.5 * total_C, 8), constrained_layout=True)
    fig1.suptitle(f'Feature Map Visualization (first {total_C} channels, SNR={client.channel.snr_db:.1f}dB)', fontsize=13)

    for row_idx, (arr, label, col) in enumerate(zip(arrays, labels, colors)):
        for ch in range(total_C):
            ax = axes1[row_idx, ch]
            im = ax.imshow(arr[ch], cmap='viridis', aspect='auto')
            if ch == 0:
                ax.set_ylabel(label, fontsize=10, color=col, fontweight='bold')
            ax.set_title(f'ch {ch}', fontsize=8)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(col)
                spine.set_linewidth(2.5)
                spine.set_visible(True)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()

print("\n=== 去噪還原特徵圖 ===")
visualize_activation_denoising(clients[0], main_server, test_loader, device)

print("\n=== Visualizing Feature Map ===")
visualize_feature_maps(clients[0], test_loader, device, num_channels=8)

print(f"Finished!")
