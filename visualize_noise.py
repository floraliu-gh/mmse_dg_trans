"""
視覺化通道雜訊的影響
展示不同 SNR 下訊號的劣化程度
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from channel_simulation import CommunicationChannel, PixelNoiseInjector
from torchvision import datasets, transforms


def visualize_channel_effects():
    """
    視覺化通道對訊號的影響
    """
    # 產生測試訊號
    t = np.linspace(0, 2*np.pi, 1000)
    clean_signal = np.sin(t) + 0.5 * np.sin(3*t)
    
    # 轉換為 tensor
    clean_tensor = torch.tensor(clean_signal, dtype=torch.float32).unsqueeze(0)
    
    # 不同 SNR
    snr_values = [20, 15, 10, 5]
    
    fig, axes = plt.subplots(len(snr_values) + 1, 1, figsize=(12, 10))
    
    # 原始訊號
    axes[0].plot(t, clean_signal, 'b-', linewidth=2)
    axes[0].set_title('Clean Signal (Original)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-2, 2])
    
    # 不同 SNR 的雜訊訊號
    for i, snr in enumerate(snr_values):
        channel = CommunicationChannel(snr_db=snr, channel_gain=1.0, bit_error_rate=0.0)
        noisy_tensor = channel.transmit(clean_tensor, add_bit_error=False)
        noisy_signal = noisy_tensor.squeeze().numpy()
        
        axes[i+1].plot(t, clean_signal, 'b-', alpha=0.3, linewidth=1, label='Clean')
        axes[i+1].plot(t, noisy_signal, 'r-', linewidth=1.5, label='Noisy')
        axes[i+1].set_title(f'SNR = {snr} dB', fontsize=12, fontweight='bold')
        axes[i+1].set_ylabel('Amplitude')
        axes[i+1].grid(True, alpha=0.3)
        axes[i+1].legend(loc='upper right')
        axes[i+1].set_ylim([-2, 2])
    
    axes[-1].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('channel_noise_visualization.png', dpi=150)
    print("通道雜訊視覺化已儲存: channel_noise_visualization.png")
    plt.close()


def visualize_pixel_noise():
    """
    視覺化像素雜訊對影像的影響
    """
    # 載入一張 EuroSAT 影像
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.EuroSAT('./data', download=True, transform=transform)
    
    # 取一張影像
    image, label = dataset[100]
    
    # 反正規化以便顯示
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = image * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)
    
    # 不同程度的像素雜訊
    noise_levels = [0.0, 0.05, 0.1, 0.2]
    
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(16, 4))
    
    for i, noise_std in enumerate(noise_levels):
        if noise_std == 0.0:
            noisy_image = image_denorm
            title = 'Clean Image'
        else:
            injector = PixelNoiseInjector(noise_std=noise_std)
            noisy = injector.add_noise(image)
            # 反正規化
            noisy_image = noisy * std + mean
            noisy_image = torch.clamp(noisy_image, 0, 1)
            title = f'Noise std={noise_std}'
        
        # 轉換為 numpy 並調整維度順序
        img_np = noisy_image.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('pixel_noise_visualization.png', dpi=150)
    print("像素雜訊視覺化已儲存: pixel_noise_visualization.png")
    plt.close()


def visualize_bit_errors():
    """
    視覺化位元錯誤的影響
    """
    # 產生測試資料
    clean_data = torch.randn(1, 256) * 5  # 模擬 smashed data
    
    # 不同的位元錯誤率
    ber_values = [0.0, 0.001, 0.005, 0.01]
    
    fig, axes = plt.subplots(len(ber_values), 1, figsize=(12, 10))
    
    for i, ber in enumerate(ber_values):
        channel = CommunicationChannel(snr_db=999, channel_gain=1.0, bit_error_rate=ber)
        
        if ber == 0.0:
            corrupted_data = clean_data
        else:
            corrupted_data = channel.add_bit_errors(clean_data, num_bits=8)
        
        # 計算誤差
        error = (corrupted_data - clean_data).squeeze().numpy()
        
        axes[i].plot(clean_data.squeeze().numpy(), 'b-', alpha=0.5, label='Clean', linewidth=1)
        axes[i].plot(corrupted_data.squeeze().numpy(), 'r-', label='With Bit Errors', linewidth=1)
        axes[i].set_title(f'Bit Error Rate = {ber}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # 顯示誤差統計
        mse = np.mean(error ** 2)
        axes[i].text(0.02, 0.98, f'MSE: {mse:.4f}', 
                    transform=axes[i].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Feature Index')
    
    plt.tight_layout()
    plt.savefig('bit_error_visualization.png', dpi=150)
    print("位元錯誤視覺化已儲存: bit_error_visualization.png")
    plt.close()


def compare_denoising():
    """
    比較去雜訊前後的效果
    """
    from channel_simulation import Denoiser
    
    # 產生測試訊號
    t = np.linspace(0, 2*np.pi, 1000)
    clean_signal = np.sin(t) + 0.5 * np.sin(3*t)
    clean_tensor = torch.tensor(clean_signal, dtype=torch.float32).unsqueeze(0)
    
    # 加入雜訊
    channel = CommunicationChannel(snr_db=10, channel_gain=1.0, bit_error_rate=0.0)
    noisy_tensor = channel.transmit(clean_tensor, add_bit_error=False)
    
    # 去雜訊
    denoiser = Denoiser(alpha=0.3, method='ema')
    
    # 逐步去雜訊
    denoised_signals = []
    current_signal = noisy_tensor.clone()
    
    for _ in range(10):  # 10 次迭代
        current_signal = denoiser.denoise(current_signal)
        denoised_signals.append(current_signal.squeeze().numpy())
    
    # 視覺化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 原始 vs 雜訊
    axes[0, 0].plot(t, clean_signal, 'b-', label='Clean', linewidth=2)
    axes[0, 0].plot(t, noisy_tensor.squeeze().numpy(), 'r-', alpha=0.6, label='Noisy (SNR=10dB)', linewidth=1.5)
    axes[0, 0].set_title('Original vs Noisy Signal', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 去雜訊效果
    axes[0, 1].plot(t, clean_signal, 'b-', label='Clean', linewidth=2)
    axes[0, 1].plot(t, noisy_tensor.squeeze().numpy(), 'r-', alpha=0.3, label='Noisy', linewidth=1)
    axes[0, 1].plot(t, denoised_signals[-1], 'g-', label='Denoised', linewidth=2)
    axes[0, 1].set_title('Denoising Result', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 誤差曲線
    mse_noisy = np.mean((noisy_tensor.squeeze().numpy() - clean_signal) ** 2)
    mse_denoised = [np.mean((sig - clean_signal) ** 2) for sig in denoised_signals]
    
    axes[1, 0].plot([mse_noisy] * len(mse_denoised), 'r--', label='Noisy MSE', linewidth=2)
    axes[1, 0].plot(mse_denoised, 'g-', label='Denoised MSE', linewidth=2, marker='o')
    axes[1, 0].set_title('MSE Reduction over Iterations', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Mean Squared Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 改善率
    improvement = [(mse_noisy - mse) / mse_noisy * 100 for mse in mse_denoised]
    axes[1, 1].plot(improvement, 'b-', linewidth=2, marker='s')
    axes[1, 1].set_title('Noise Reduction (% Improvement)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('denoising_comparison.png', dpi=150)
    print("去雜訊效果視覺化已儲存: denoising_comparison.png")
    plt.close()


if __name__ == '__main__':
    print("視覺化已停用。")
    # print("生成視覺化圖表...")
    # print("\n1. 通道雜訊效果")
    # visualize_channel_effects()
    
    # print("\n2. 像素雜訊效果")
    # visualize_pixel_noise()
    
    # print("\n3. 位元錯誤效果")
    # visualize_bit_errors()
    
    # print("\n4. 去雜訊效果比較")
    # compare_denoising()
    
    # print("\n所有視覺化完成!")
