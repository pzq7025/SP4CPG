import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, SequentialSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm

# from preprocess.graph_dataset import CPGDataset
from models.gnn_models import GCN, GAT, GIN, GraphSAGE, GGNN  # 可替换为 GAT, GIN 等
from models.hgcn_models import HGCN  # 如果需要使用HGNN模型
from tabulate import tabulate


# ========== TensorBoard ==========
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# ========== 日志文件 ==========
timestamp = time.time()
formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M")
log_file = os.path.join(log_dir, f"train_log_{formatted_time}.txt")
f_log = open(log_file, "w")

# ========== 初始化 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# 若使用GPU，输出详细信息
if device.type == 'cuda':
    #torch.cuda.set_device(device)  # 设置当前使用的 GPU
    # 构建表格数据
    device_table = [
        ["GPU Num", f"{torch.cuda.device_count()}"],
        ["GPU Index In USE", f"{torch.cuda.current_device()}"],
        ["GPU Name", f"{torch.cuda.get_device_name(device)}"],
        ["GPU Memory", f"{torch.cuda.get_device_properties(device).total_memory / 1024 / 1024 / 1024:.0f} GB"],
        # ["CUDA Version", f"{torch.version.cuda}"]
    ]
    # print("=============== Device Details ===============")
    # print(tabulate(device_table, headers=["Items", "Value"], tablefmt="grid") + "\n")
    f_log.write("=============== Device Details ===============\n")
    f_log.write(tabulate(device_table, headers=["Items", "Value"], tablefmt="grid") + "\n\n")
    print(f"\nDevice Details: GPU Index In USE:{torch.cuda.current_device()}, GPU Name: {torch.cuda.get_device_name(device)}, GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 / 1024 / 1024:.0f} GB, CUDA Version: {torch.version.cuda}")

else:
    # CPU 训练提示
    import platform
    print("⚠️ 注意: 当前使用 CPU 进行训练, 训练速度可能较慢.")
    print("💡 提示: 若有可用 GPU, 可加速训练。请确保正确安装 PyTorch GPU 版本.")
    print(f"CPU信息: {platform.processor()}")

# ========== 参数设置 ==========
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=GGNN, type=str, help="or GAT, GIN, GraphSAGE, GGNN", required=True)
parser.add_argument("--batch", default=128, type=int, help="batchsize")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")
parser.add_argument("--dropout", default=0.4, type=float, help="dropout rate")
parser.add_argument("--epoch", default=500, type=int, help="train epochs")
parser.add_argument("--patience", default=100, type=int, help="early stopping")
args = parser.parse_args()

# 构建表格数据
param_table = [
    ["Model", f"{args.model}"],
    ["Batchsize", f"{args.batch}"],
    ["Learning_Rate", f"{args.lr}"],
    ["Dropout", f"{args.dropout}"],
    ["Epochs", f"{args.epoch}"],
    ["Early Stopping", f"{args.patience}"]
]
f_log.write("====== Param Details ======\n")
f_log.write(tabulate(param_table, headers=["Params", "Value"], tablefmt="grid") + "\n")
# print("====== Params Details ======")
# print(tabulate(param_table, headers=["Items", "Value"], tablefmt="grid"))
print(f"\nParams Details: Model={args.model}, Batchsize={args.batch}, Learning_Rate={args.lr}, Dropout={args.dropout}, Epochs={args.epoch}, Early Stopping={args.patience}")

# ========== 训练函数 ==========
def train(model, train_batch, optimizer, epoch):
    model.train()
    data = train_batch.to(device)
    optimizer.zero_grad()
    out = model(data)

    target = data.y.view(-1).long()
    loss = F.cross_entropy(out, target)
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    correct = int((pred == target).sum().item())
    bs = target.size(0)

    # 把 mean loss 还原成 sum loss，方便 epoch 级别正确平均
    return correct, loss.item() * bs, bs


# ========== 验证函数 ==========
def evaluate(model, val_batch, epoch):
    """Evaluate the model on the validation set."""
    model.eval()
    correct = 0
    total_samples = 0
    with torch.no_grad():
        # for data in tqdm(loader, desc=f"Epoch {epoch:03d}/{args.epoch} - Validation", leave=False, ncols=80):
        # for data in loader:
        data = val_batch.to(device)
        out = model(data)
        
        # 确保标签维度正确
        target = data.y.view(-1)
        
        pred = out.argmax(dim=1)
        correct += int((pred == target).sum())
        total_samples += target.size(0)
            
    # return correct / total_samples
    return correct

# ========== 最终评估 ==========
def evaluate_metrics(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            prob = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob[:, 1].cpu().numpy() if prob.shape[1] > 1 else prob[:, 0].cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

    return acc, prec, rec, f1, fpr, fnr, auc, cm


# ========== 加载数据集 ==========
# dataset = CPGDataset(root="preprocess")
dataset = torch.load('./preprocess/hcpg_dataset.pkl', weights_only=False)
train_len = int(0.8 * len(dataset))
val_len = int(0.1 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
# print(f"Dataset size: {len(dataset)}")
f_log.write(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

train_sampler = RandomSampler(train_dataset)
val_sampler = SequentialSampler(val_dataset)    # 使用 SequentialSampler 确保数据顺序一致
test_sampler = SequentialSampler(test_dataset)  # 使用 SequentialSampler 确保数据顺序一致

train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch, num_workers=0)
val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch, num_workers=0)
test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch, num_workers=0)

# print(f'val_loader: {len(val_loader)}, test_loader: {len(test_loader)}')
# print(f"\nDataset Splits: Train={train_len}, Val={val_len}, Test={test_len}")

# ========== 加载模型 ==========
if args.model == 'GCN':
    model = GCN(args.dropout, in_channels=768, hidden_channels=512).to(device)
elif args.model == 'HGCN':
    model = HGCN(args.dropout, in_channels=768, hidden_channels=512).to(device)
elif args.model == 'GAT':
    model = GAT(args.dropout, in_channels=768, hidden_channels=512).to(device)
elif args.model == 'GGNN':
    model = GGNN(args.dropout, in_channels=768, out_channels=512).to(device)
elif args.model == 'GIN':
    model = GIN(args.dropout, in_channels=768, hidden_channels=512).to(device)
elif args.model == 'GraphSAGE':
    model = GraphSAGE(args.dropout, in_channels=768, hidden_channels=512).to(device)
# elif args.model == 'HGNN':
#     model = HGNN(dropout=0.5, in_channels=768, hidden_channels=128, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# ========== Early Stopping ==========
patience = args.patience
best_val_acc = 0.0
best_model_state = None
no_improve_epochs = 0

# ========== 主训练循环 ==========
start_total = time.time()
final_epoch = 0

print("\n================================== Start Training ==================================")
f_log.write("\n================================== Start Training ==================================\n")
for epoch in range(args.epoch):
    epoch_start = time.time()   # 记录每个epoch开始时间
    # ------训练开始（Training Start）------
    train_correct = 0
    loss_sum = 0
    train_samples = 0
    train_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epoch} - Training", ncols=80, leave=False)
    for idx, batch in enumerate(train_bar, start=1):    # 以batch为单位进行训练
        idx_correct, idx_loss, bs = train(model, batch, optimizer, epoch)
        train_correct += idx_correct
        loss_sum += idx_loss
        train_samples += bs
    train_acc = train_correct / train_samples # 计算每个batch的准确率
    train_loss = loss_sum / train_samples   # 计算每个batch的平均损失
    # ------训练结束（Training End）------

    # ------验证开始（Validation Start）------
    val_correct = 0
    val_bar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{args.epoch} - Validation", ncols=80, leave=False)
    for idx, batch in enumerate(val_bar, start=1):  # 以batch为单位进行验证
        idx_correct = evaluate(model, batch, epoch)
        val_correct += idx_correct
    val_acc = val_correct / val_len  # 计算每个batch的准确率
    # ------验证结束（Validation End）------
    epoch_time = time.time() - epoch_start

    # TensorBoard记录
    writer.add_scalar('Acc/train', train_acc, epoch)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Acc/val', val_acc, epoch)
    writer.add_scalar('Time/train_epoch_sec', epoch_time, epoch)

    # 文本日志记录
    log_str = f"Epoch: {epoch:03d}/{args.epoch} | Train Acc: {train_acc:.4f} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f} (s)"
    print(log_str)
    f_log.write(log_str + "\n")

    # Early stopping & 模型保存
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()
        torch.save(best_model_state, os.path.join(log_dir, f"{args.model}_{args.lr}_{args.dropout}_best_model.pt"))
        no_improve_epochs = 0
        print(f"🎉 New Best Val Acc: {best_val_acc:.4f}")
        f_log.write(f"🎉 New Best Val Acc: {best_val_acc:.4f}\n")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"⏹️  No Improvement at epoch {[epoch-args.patience, epoch]}, Early stopping!")
            f_log.write(f"⏹️  No Improvement at epoch {[epoch-args.patience, epoch]}, Early stopping!\n")
            final_epoch = epoch
            break

    final_epoch = epoch

total_time = time.time() - start_total
print(">>> Total Training Time:", f"{total_time:.2f} (s)")
f_log.write(f">>> Total Training Time: {total_time:.2f} (s)\n")
print("================================= End Training ==================================")
f_log.write("================================= End Training =================================\n")

# 加载最佳模型
print("Loading best model for final evaluation...")
model.load_state_dict(torch.load(os.path.join(log_dir, f"{args.model}_{args.lr}_{args.dropout}_best_model.pt")))

# 测试
test_start = time.time()
acc, prec, rec, f1, fpr, fnr, auc, cm = evaluate_metrics(model, test_loader)
test_time = time.time() - test_start


# 构建表格数据
metrics_table = [
    ["Test Accuracy", f"{acc:.4f}"],
    ["Precision", f"{prec:.4f}"],
    ["Recall", f"{rec:.4f}"],
    ["F1 Score", f"{f1:.4f}"],
    ["False Positive Rate (FPR)", f"{fpr:.4f}"],
    ["False Negative Rate (FNR)", f"{fnr:.4f}"],
    ["AUC", f"{auc:.4f}"],
    ["Test Evaluation Time (s)", f"{test_time:.2f}"]
]

# 打印表格输出结果
print("\n========== Evaluation Results ==========")
f_log.write("\n========== Evaluation Results ==========\n")
print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))
f_log.write(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

# 打印 Confusion Matrix
# print("\n======= Confusion Matrix =======")
# f_log.write("\n======= Confusion Matrix =======\n")
# print(tabulate(cm, headers=["Pred 0", "Pred 1"], showindex=["True 0", "True 1"], tablefmt="grid"))
# f_log.write(tabulate(cm, headers=["Pred 0", "Pred 1"], showindex=["True 0", "True 1"], tablefmt="grid"))

f_log.write("\n\n🎯 Training and Testing completed successfully! Check logs directory for detailed results.")
print("🎯 Training and Testing completed successfully! Check logs directory for detailed results.")

f_log.close()
writer.close()