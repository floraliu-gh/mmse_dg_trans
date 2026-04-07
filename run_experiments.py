"""
實驗腳本: 比較不同通訊條件下的 SplitFed 性能
測試項目:
1. 不同 SNR 下的表現
2. 不同聚合方法的魯棒性
3. 去雜訊的效果
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json

from client_model import ClientModel
from server_model import ServerModel
from client1 import Client    
from server1 import MainServer  
from robust_aggregation import fedserver
from channel_simulation import CommunicationChannel, PixelNoiseInjector, Denoiser


def run_experiment(config):
    """
    執行單次實驗
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 解析設定
    K = config['num_clients']
    rounds = config['rounds']
    local_epochs = config['local_epochs']
    batch_size = config['batch_size']
    lr = config['lr']
    
    snr_db = config['snr_db']
    ber = config['bit_error_rate']
    enable_denoising = config['enable_denoising']
    aggregation_method = config['aggregation_method']
    
    print(f"\n{'='*60}")
    print(f"實驗: SNR={snr_db}dB, BER={ber}, Agg={aggregation_method}, Denoise={enable_denoising}")
    print(f"{'='*60}")
    
    # 資料準備
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.EuroSAT('./data', download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset_full, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定隨機種子
    )
    
    nk_list = [len(train_dataset_full) // K] * K
    nk_list[-1] += len(train_dataset_full) - sum(nk_list)
    client_subsets = random_split(train_dataset_full, nk_list)
    
    dataloaders = [DataLoader(sub, batch_size=batch_size, shuffle=True) for sub in client_subsets]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化通訊模組
    channels = []
    for i in range(K):
        if snr_db is not None:
            channel = CommunicationChannel(snr_db=snr_db, channel_gain=1.0, bit_error_rate=ber)
        else:
            channel = None
        channels.append(channel)
    
    denoiser = Denoiser(alpha=0.3, method='ema') if enable_denoising else None
    
    # 初始化模型
    global_server_model = ServerModel().to(device)
    main_server = MainServer(global_server_model, device, lr=lr, denoiser=denoiser)
    
    clients = []
    for i in range(K):
        c_model = ClientModel().to(device)
        c_instance = Client(
            model=c_model, 
            data_loader=dataloaders[i], 
            device=device, 
            lr=lr,
            channel=channels[i],
            pixel_noise_injector=None
        )
        clients.append(c_instance)
    
    # 訓練
    test_accuracies = []
    
    for r in range(rounds):
        round_loss = 0.0
        total_steps = 0
        
        for i in range(K):
            steps_per_client = len(dataloaders[i]) * local_epochs
            
            for _ in range(steps_per_client):
                A_k_received, y = clients[i].ClientUpdate(add_pixel_noise=False)
                dA_k, loss_val = main_server.ServerUpdate(A_k_received, y)
                main_server.step()
                clients[i].ClientBackprop(dA_k)
                
                round_loss += loss_val
                total_steps += 1
        
        # 聚合
        client_models_list = [c.model for c in clients]
        aggregated_model = fedserver(
            client_models_list, nk_list, sum(nk_list),
            method=aggregation_method, trim_ratio=0.2, f=1
        )
        
        global_weights = aggregated_model.state_dict()
        for c in clients:
            c.model.load_state_dict(global_weights)
        
        if enable_denoising and denoiser is not None:
            denoiser.reset()
        
        # 評估
        clients[0].model.eval()
        main_server.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                activations = clients[0].model(x)
                outputs = main_server.model(activations)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        test_acc = correct / total
        test_accuracies.append(test_acc)
        
        if (r + 1) % 5 == 0:
            print(f"Round {r+1}/{rounds}: Acc = {test_acc*100:.2f}%")
    
    return {
        'config': config,
        'test_accuracies': test_accuracies,
        'final_accuracy': test_accuracies[-1],
        'best_accuracy': max(test_accuracies)
    }


def main():
    """
    主實驗流程
    """
    # 基礎設定
    base_config = {
        'num_clients': 5,
        'rounds': 15,
        'local_epochs': 1,
        'batch_size': 128,
        'lr': 0.001
    }
    
    # 實驗 1: 測試不同 SNR 的影響
    print("\n" + "="*80)
    print("實驗 1: 不同 SNR 對性能的影響")
    print("="*80)
    
    snr_experiments = []
    snr_values = [None, 20, 15, 10, 5]  # None = 無雜訊
    
    for snr in snr_values:
        config = base_config.copy()
        config['snr_db'] = snr
        config['bit_error_rate'] = 0.0
        config['enable_denoising'] = False
        config['aggregation_method'] = 'fedavg'
        
        result = run_experiment(config)
        snr_experiments.append(result)
    
    # 實驗 2: 測試不同聚合方法
    print("\n" + "="*80)
    print("實驗 2: 不同聚合方法在雜訊環境下的魯棒性")
    print("="*80)
    
    agg_experiments = []
    agg_methods = ['fedavg', 'median', 'trimmed_mean']
    
    for method in agg_methods:
        config = base_config.copy()
        config['snr_db'] = 10  # 中等雜訊
        config['bit_error_rate'] = 0.001
        config['enable_denoising'] = False
        config['aggregation_method'] = method
        
        result = run_experiment(config)
        agg_experiments.append(result)
    
    # 實驗 3: 測試去雜訊的效果
    print("\n" + "="*80)
    print("實驗 3: 去雜訊器的效果")
    print("="*80)
    
    denoise_experiments = []
    
    for enable_denoise in [False, True]:
        config = base_config.copy()
        config['snr_db'] = 10
        config['bit_error_rate'] = 0.001
        config['enable_denoising'] = enable_denoise
        config['aggregation_method'] = 'trimmed_mean'
        
        result = run_experiment(config)
        denoise_experiments.append(result)
    
    # 視覺化因為您的要求已關閉
    # (原本這裡是產出實驗比較圖表的程式碼)
    print("\n[提示] 視覺化畫圖功能已關閉。")
    
    # 輸出數值結果
    print("\n" + "="*80)
    print("實驗結果總結")
    print("="*80)
    
    print("\n1. SNR 影響:")
    for result in snr_experiments:
        snr = result['config']['snr_db']
        label = f"SNR={snr}dB" if snr is not None else "No Noise"
        print(f"  {label:15s}: Final={result['final_accuracy']*100:.2f}%, "
              f"Best={result['best_accuracy']*100:.2f}%")
    
    print("\n2. 聚合方法比較 (SNR=10dB):")
    for result in agg_experiments:
        method = result['config']['aggregation_method']
        print(f"  {method:15s}: Final={result['final_accuracy']*100:.2f}%, "
              f"Best={result['best_accuracy']*100:.2f}%")
    
    print("\n3. 去雜訊效果 (SNR=10dB):")
    for result in denoise_experiments:
        denoise = "With Denoise" if result['config']['enable_denoising'] else "Without Denoise"
        print(f"  {denoise:15s}: Final={result['final_accuracy']*100:.2f}%, "
              f"Best={result['best_accuracy']*100:.2f}%")
    
    # 儲存結果為 JSON
    all_results = {
        'snr_experiments': snr_experiments,
        'agg_experiments': agg_experiments,
        'denoise_experiments': denoise_experiments
    }
    
    # 轉換 numpy 為 Python 原生型別
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    all_results = convert_to_serializable(all_results)
    
    with open('experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n結果已儲存: experiment_results.json")


if __name__ == '__main__':
    main()
