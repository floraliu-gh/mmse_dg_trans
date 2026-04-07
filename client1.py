import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, model, device, lr=1e-3, channel=None, pixel_noise_injector=None):
        self.model = model.to(device)
        self.device = device
        # 建議：如果使用 SGD，lr 建議設大一點 (e.g., 0.01)；Adam 則維持 1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        self.last_A_k = None
        
        self.channel = channel
        self.pixel_noise_injector = pixel_noise_injector

    def ClientUpdate(self, x, add_pixel_noise=False):
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. 確保資料在正確的裝置
        x = x.to(self.device)
        
        # 2. (可選) 在像素層面加入雜訊
        if add_pixel_noise and self.pixel_noise_injector is not None:
            # 確保雜訊不會破壞 requires_grad (雖然 input 通常不需要 grad)
            x = self.pixel_noise_injector.add_noise(x)

        # 3. 前向傳播
        # 這裡會建立 Computational Graph
        A_k = self.model(x)
        
        # 4. 【關鍵】存下這個變數，它是整張圖的「葉子」，等一下反向傳播要用
        # 注意：千萬不能在這裡 detach，否則梯度會斷掉
        self.last_A_k = A_k 
        
        # --- 新增：語意特徵量化 (Semantic Feature Quantization) ---
        # 模擬將 32-bit 浮點數壓縮為 16-bit 浮點數 (Half-Precision)
        # 節省 50% 通訊頻寬，且對模型準確率幾乎沒有影響
        A_k_detached = A_k.detach()
        
        # 轉換為 float16 (量化)
        A_k_fp16 = A_k_detached.half()
        
        # 轉換回 float32 (模擬接收端還原，因為 Server 的神經網路吃 float32)
        A_k_dequant = A_k_fp16.float()
        
        # 5. 準備傳送給 Server 的數據
        # 送出經過量化再還原的特徵 (模擬實際傳輸 payload 只有 Int8 的情況)
        if self.channel is not None:
            A_k_transmitted = self.channel.transmit(A_k_dequant, add_awgn=True, add_bit_error=False)
        else:
            A_k_transmitted = A_k_dequant

        return A_k_transmitted.clone()

    def ClientBackprop(self, dA_k):
        """
        dA_k: 從 Server 傳回來的梯度 (Gradient of Loss w.r.t Activation)
        """
        if dA_k is None:
            return

        # 1. 確保梯度在正確的裝置
        dA_k = dA_k.to(self.device)
        
        # 2. 模擬下行鏈路雜訊 (Server -> Client)
        # 通常 Downlink 頻寬較大且較穩，但若要模擬也可以加
        if self.channel is not None:
            # 這裡 add_bit_error 通常設 False，除非你想模擬極端惡劣環境
            dA_k = self.channel.transmit(dA_k, add_bit_error=False)

        # 3. 反向傳播
        # 對算出來的 Activation (last_A_k) 進行微分，dA_k 是鏈式法則中上一層傳來的梯度
        self.last_A_k.backward(dA_k)
        
        # === 梯度裁剪 ===
        #這行代碼會檢查所有參數的梯度，如果總長度超過 10，就把它們等比例縮小
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 4. 更新權重
        self.optimizer.step()
        self.last_A_k = None