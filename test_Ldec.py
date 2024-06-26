import torch
import torch.nn.functional as F

# 假設 logits_clean 和 logits_noisy 的形狀是 [batch_size, seq_len, num_classes]
batch_size = 2
seq_len = 3
num_classes = 4

# 隨機生成 logits_clean 和 logits_noisy 作為示例
logits_clean = torch.randn(batch_size, seq_len, num_classes)
# print("logits_clean :", logits_clean)
# print("logits_clean's shape :", logits_clean.shape)
logits_noisy = torch.randn(batch_size, seq_len, num_classes)
# print("logits_noisy :", logits_noisy)
# print("logits_noisy's shape :", logits_noisy.shape)

def test_l_dec(logits_clean, logits_noisy):
    # 修改 logits_clean 以進行 one-hot 編碼並計算 target class
    target_class = torch.argmax(logits_clean, dim=-1)
    # print("Target class indices (from logits_clean):", target_class)
    # print("Target class indices (from logits_clean) shape:", target_class.shape)
    
    # 計算 logits_noisy 的概率
    logits_noisy_exp = torch.exp(logits_noisy)
    # print("Exponentiated logits_noisy:", logits_noisy_exp)
    # print("Exponentiated logits_noisy's shape:", logits_noisy_exp.shape)

    # 提取目標類別位置的預測值
    target_logits_noisy_exp = logits_noisy_exp.gather(2, target_class.unsqueeze(-1)).squeeze(-1)
    # print("Target class logits (from logits_noisy_exp):", target_logits_noisy_exp)
    # print("Target class logits (from logits_noisy) shape:", target_logits_noisy_exp.shape)

    # 計算所有類別預測值的和
    sum_logits_noisy_exp = logits_noisy_exp.sum(dim=-1)
    # print("Sum of all class logits (from logits_noisy_exp):", sum_logits_noisy_exp)

    # 計算 Loss_dec
    Loss_dec = -torch.log(target_logits_noisy_exp / sum_logits_noisy_exp).mean()
    print("Calculated Loss_dec:", Loss_dec)

def test_l_dec_ce(logits_clean, logits_noisy):

    # 修改 logits_clean 以進行 one-hot 編碼並計算 target class
    target_class = torch.argmax(logits_clean, dim=-1)
    print("target_class :", target_class)
    # 將 logits_noisy 轉換為 [batch_size * seq_len, num_classes] 的形狀
    # print("logits noisy before flat :", logits_noisy)
    # print("logits noisy before flat shape :", logits_noisy.shape)
    logits_noisy_flat = logits_noisy.view(-1, num_classes)
    print("logits noisy after flat :", logits_noisy_flat)
    # print("logits noisy after flat shape:", logits_noisy_flat.shape)
    # 將 target_class 轉換為 [batch_size * seq_len] 的形狀
    target_class_flat = target_class.view(-1)
    print("target_class_flat :", target_class_flat)
    
    # 計算 cross entropy loss
    Loss_dec = F.cross_entropy(logits_noisy_flat, target_class_flat)
    print("Calculated Loss_dec_ce:", Loss_dec)

    
# 執行測試函數
test_l_dec(logits_clean, logits_noisy)
test_l_dec_ce(logits_clean, logits_noisy)
