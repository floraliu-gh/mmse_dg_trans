import torch
import torch.nn as nn
import torch.optim as optim

class MainServer:
    def __init__(self, model, device, lr=0.01, denoiser=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=6)
        
        # 去雜訊器
        self.denoiser = denoiser
    
    def ServerUpdate(self, A_k, y, clear_grad=True):
        """
        Server 的 forward + backward
        """
        if clear_grad:
            self.optimizer.zero_grad()

        # 1. 統一將數據移動到 Device
        # 注意：我們使用一個新變數 input_tensor 來承接，避免混淆
        input_tensor = A_k.to(self.device)
        y = y.to(self.device)
        
        # 2. 去雜訊處理邏輯
        if self.denoiser is not None:
            # 如果去雜訊，input_tensor 會變成一個計算過的新 Tensor
            input_tensor = self.denoiser.denoise(input_tensor)
        
        # 3. 【關鍵修正】強制設定它為葉子節點 (Leaf Tensor)
        # 無論是否經過 Denoise，我們都把「準備送進 Model 的這個 Tensor」視為起點
        # .detach() 切斷與前面的關聯，.requires_grad_(True) 開啟新的追蹤
        input_tensor = input_tensor.detach().requires_grad_(True)

        # 4. Forward pass (使用處理過的 input_tensor)
        y_hat = self.model(input_tensor)
        loss = self.criterion(y_hat, y)

        # 5. Backward pass
        loss.backward()

        # 6. 取得梯度
        # 因為我們在第 3 步強制宣告了 requires_grad，所以這裡一定會有 grad
        if input_tensor.grad is None:
            # 防呆機制：如果萬一還是 None，回傳全 0 梯度避免程式崩潰
            dA_k = torch.zeros_like(input_tensor)
        else:
            dA_k = input_tensor.grad.detach()
            
        return dA_k, loss.item()

    def step(self):
        self.optimizer.step()
