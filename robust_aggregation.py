import copy
import torch

def fedavg(client_models, nk_list, n):
    """
    標準 FedAvg: 加權平均聚合
    """
    global_model = copy.deepcopy(client_models[0])
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])

    # 加權平均
    for k, client_model in enumerate(client_models):
        client_dict = client_model.state_dict()
        weight = nk_list[k] / n
       
        for key in global_dict.keys():
            if global_dict[key].dtype in [torch.float32, torch.float64, torch.float16]:
                global_dict[key] += weight * client_dict[key]
            else:
                global_dict[key] = client_dict[key]

    global_model.load_state_dict(global_dict)
    return global_model


def fed_median(client_models):
    """
    中位數聚合: 對每個參數取中位數
    優點: 對異常值(outliers)更魯棒,適合有雜訊的環境
    """
    global_model = copy.deepcopy(client_models[0])
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        if global_dict[key].dtype in [torch.float32, torch.float64, torch.float16]:
            # 收集所有 client 的參數
            param_list = [client.state_dict()[key] for client in client_models]
            param_tensor = torch.stack(param_list, dim=0)
            
            # 取中位數
            global_dict[key] = torch.median(param_tensor, dim=0)[0]
        else:
            # 非浮點數參數直接取第一個
            global_dict[key] = client_models[0].state_dict()[key]
    
    global_model.load_state_dict(global_dict)
    return global_model


def fed_trimmed_mean(client_models, nk_list, n, trim_ratio=0.2):
    """
    修剪平均聚合: 去掉最大和最小的極端值後平均
    參數:
    - trim_ratio: 要修剪的比例 (0.0-0.5)
    """
    K = len(client_models)
    trim_count = int(K * trim_ratio)
    
    global_model = copy.deepcopy(client_models[0])
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        if global_dict[key].dtype in [torch.float32, torch.float64, torch.float16]:
            # 收集所有參數
            param_list = [client.state_dict()[key] for client in client_models]
            param_tensor = torch.stack(param_list, dim=0)
            
            # 排序並修剪
            sorted_params, _ = torch.sort(param_tensor, dim=0)
            
            if trim_count > 0:
                # 去掉最小和最大的值
                trimmed = sorted_params[trim_count:-trim_count]
            else:
                trimmed = sorted_params
            
            # 平均
            global_dict[key] = torch.mean(trimmed, dim=0)
        else:
            global_dict[key] = client_models[0].state_dict()[key]
    
    global_model.load_state_dict(global_dict)
    return global_model


def fed_krum(client_models, f=1):
    """
    Krum 聚合: 選擇與其他模型最接近的一個
    參數:
    - f: 容許的拜占庭節點數量
    適合有惡意節點的情況
    """
    K = len(client_models)
    
    # 計算每對模型之間的距離
    distances = torch.zeros(K, K)
    
    for i in range(K):
        for j in range(i+1, K):
            dist = 0.0
            dict_i = client_models[i].state_dict()
            dict_j = client_models[j].state_dict()
            
            for key in dict_i.keys():
                if dict_i[key].dtype in [torch.float32, torch.float64, torch.float16]:
                    dist += torch.sum((dict_i[key] - dict_j[key]) ** 2).item()
            
            distances[i, j] = dist
            distances[j, i] = dist
    
    # 對每個模型,計算到最近 K-f-2 個模型的距離總和
    scores = torch.zeros(K)
    for i in range(K):
        sorted_dists, _ = torch.sort(distances[i])
        # 排除自己(距離=0)和最遠的 f+1 個
        scores[i] = torch.sum(sorted_dists[1:K-f-1])
    
    # 選擇得分最低的模型(最接近其他模型)
    selected_idx = torch.argmin(scores).item()
    
    return copy.deepcopy(client_models[selected_idx])


def fedserver(client_models, nk_list, n, method='fedavg', **kwargs):
    """
    統一的聚合介面
    參數:
    - method: 'fedavg', 'median', 'trimmed_mean', 'krum'
    """
    if method == 'fedavg':
        return fedavg(client_models, nk_list, n)
    elif method == 'median':
        return fed_median(client_models)
    elif method == 'trimmed_mean':
        trim_ratio = kwargs.get('trim_ratio', 0.2)
        return fed_trimmed_mean(client_models, nk_list, n, trim_ratio)
    elif method == 'krum':
        f = kwargs.get('f', 1)
        return fed_krum(client_models, f)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
