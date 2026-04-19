import torch
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
from channel_simulation import CommunicationChannel, PixelNoiseInjector, MMSEDenoiser

EXPERIMENT_NAME = "MMSE_Denoising"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{EXPERIMENT_NAME}] Using device: {device}")

K = 5
rounds = 50
local_epochs = 3  # 降低 local epochs，減少過擬合風險
batch_size = 128
lr = 0.002

ENABLE_CHANNEL_NOISE = True
ENABLE_PIXEL_NOISE = False

AGGREGATION_METHOD = 'fedavg'

SNR_DB = 15
CHANNEL_GAIN = 1.0
BIT_ERROR_RATE = 0.001
PIXEL_NOISE_STD = 0.05

print(f"\n=== [{EXPERIMENT_NAME}] Settings ===")
print(f"Channel noise: On (SNR={SNR_DB}dB)")
print(f"Denoising: MMSE (Wiener Filter, SNR={SNR_DB}dB)")
print(f"Aggregation method: {AGGREGATION_METHOD}")

print("\nPreparing DeepGlobe dataset...")
from deepglobe import DeepGlobeDataset

full_dataset = DeepGlobeDataset(root_dir='C:/Users/user/Desktop/Flora/deepglobe', split=None, img_size=(64, 64))
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
channel_snrs = []
for i in range(K):
    snr_variation = SNR_DB + np.random.randn() * 2
    channel_snrs.append(snr_variation)
    channel = CommunicationChannel(snr_db=snr_variation, channel_gain=CHANNEL_GAIN, bit_error_rate=BIT_ERROR_RATE)
    pixel_injector = PixelNoiseInjector(noise_std=PIXEL_NOISE_STD) if ENABLE_PIXEL_NOISE else None
    channels.append(channel)
    pixel_injectors.append(pixel_injector)

# MMSE 去雜訊器：使用平均 SNR 初始化
avg_snr_init = np.mean(channel_snrs)
mmse_denoiser = MMSEDenoiser(snr_db=avg_snr_init)
print(f"MMSE Denoiser initialized with avg SNR = {avg_snr_init:.2f} dB")

global_server_model = ServerModel(num_classes=7).to(device)
server_lr = 0.001
main_server = MainServer(global_server_model, device, lr=server_lr, denoiser=mmse_denoiser)
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
            print("目前 Batch 的真實標籤包含類別:", torch.unique(y).tolist())
            activations = c_model(x)
            outputs = s_model(activations)
            _, predicted = torch.max(outputs, 1)
            
            # mask out unknown (class 6)
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
#summary(main_server.model, input_data=(dummy_server_input,), device=device)

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

for r in range(rounds):
    # 每輪根據當前平均 SNR 更新 MMSE 的 Wiener 係數
    avg_snr = np.mean([ch.snr_db for ch in channels if ch is not None])
    mmse_denoiser.update_snr(avg_snr)

    wiener_w = mmse_denoiser.snr_linear / (mmse_denoiser.snr_linear + 1.0)
    print(f'\n--- Round {r+1}/{rounds} (MMSE SNR={avg_snr:.2f}dB, Wiener W={wiener_w:.4f}) ---')
    round_loss = 0.0
    total_steps = 0

    for i in range(K):
        for epoch in range(local_epochs):
            for batch_idx, (x, y) in enumerate(dataloaders[i]):
                x, y = x.to(device), y.to(device)

                A_k_received = clients[i].ClientUpdate(x, add_pixel_noise=ENABLE_PIXEL_NOISE)

                A_k_received = torch.nan_to_num(A_k_received, nan=0.0)
                # 移除過度正規化：不再強制 normalize activation，以保留 MMSE 去雜訊後的特徵分佈
                A_k_received = torch.clamp(A_k_received, min=-10.0, max=10.0)  # 只做寬鬆的數值保護

                A_k_server_input = A_k_received.to(device).detach().clone().requires_grad_(True)

                dA_k, loss_val = main_server.ServerUpdate(A_k_server_input, y, clear_grad=True)

                dA_k = torch.clamp(dA_k, min=-0.5, max=0.5)

                clients[i].ClientBackprop(dA_k.clone())
                torch.nn.utils.clip_grad_norm_(clients[i].model.parameters(), max_norm=5.0)

                torch.nn.utils.clip_grad_norm_(main_server.model.parameters(), max_norm=5.0)
                main_server.step()

                # DEBUG 列印區塊 (觀察梯度健康度)
                if i == 0 and batch_idx == 0:
                    grad_norm = dA_k.abs().mean().item()
                    first_layer_grad = 0
                    for param in clients[i].model.parameters():
                        if param.grad is not None:
                            first_layer_grad = param.grad.abs().mean().item()
                            break
                    print(f"Server回傳梯度={grad_norm:.6e}, Client權重梯度={first_layer_grad:.6f}")

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

# 1. 畫 Training Loss
axes[0].plot(range(1, rounds + 1), train_losses, marker='o', color='b', markersize=4, label='Train Loss')
axes[0].set_title('Training Loss over Rounds')
axes[0].set_xlabel('Communication Rounds')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# 2. 畫 Pixel Accuracy
# 將小數轉換為百分比數值方便閱讀
acc_percent = [acc * 100 for acc in test_accuracies]
axes[1].plot(range(1, rounds + 1), acc_percent, marker='s', color='g', markersize=4, label='Pixel Accuracy')
axes[1].set_title('Test Pixel Accuracy over Rounds')
axes[1].set_xlabel('Communication Rounds')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

# 3. 畫 mIoU
axes[2].plot(range(1, rounds + 1), test_mious, marker='^', color='r', markersize=4, label='mIoU')
axes[2].set_title('Test mIoU over Rounds')
axes[2].set_xlabel('Communication Rounds')
axes[2].set_ylabel('mIoU')
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].legend()
plt.show()

def visualize_activation_denoising(client, server, dataloader, device):
    client.model.eval()
    server.model.eval()

    # 取出一個 batch，並只取第一張圖
    x, y = next(iter(dataloader))
    x_single = x[0:1].to(device)

    input_image_raw = x_single.squeeze(0).cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_image_denorm = (input_image_raw * std) + mean
    input_image_vis = np.clip(input_image_denorm, 0, 1)

    with torch.no_grad():
        # 1. 原始特徵 (Client 輸出)
        clean_activations = client.model(x_single)

        # 2. 經過通道加上雜訊
        noisy_activations = client.channel.transmit(clean_activations)

        # 3. Server 端進行 MMSE 去噪
        denoised_activations = server.denoiser.denoise(noisy_activations)

    # 將特徵圖轉到 CPU，並取第一個通道 (Channel 0) 進行視覺化
    clean_vis = clean_activations[0, 0].cpu().numpy()
    noisy_vis = noisy_activations[0, 0].cpu().numpy()
    denoised_vis = denoised_activations[0, 0].cpu().numpy()

    # 建立 1x3 的畫布
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(input_image_vis)
    axes[0].set_title("Original picture")
    axes[0].axis('off')

    axes[1].imshow(clean_vis, cmap='viridis')
    axes[1].set_title("Clean Activations (Client Output)")
    axes[1].axis('off')

    axes[2].imshow(noisy_vis, cmap='viridis')
    axes[2].set_title(f"Noisy Activations (SNR={client.channel.snr_db:.1f}dB)")
    axes[2].axis('off')

    axes[3].imshow(denoised_vis, cmap='viridis')
    axes[3].set_title("Denoised Activations (MMSE)")
    axes[3].axis('off')
    plt.show()

print("\n=== Denoising plot generating ===")
visualize_activation_denoising(clients[0], main_server, test_loader, device)


print(f"Finished!")
