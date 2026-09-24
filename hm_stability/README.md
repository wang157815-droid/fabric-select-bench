# H&M 推荐基线稳定性实验

本目录提供 BPR-MF 与两层 LightGCN 的三随机种子复核代码。实验数据与原始 CSV 均未公开提交；运行时需将去标识化的 `interactions.npz` 和 `manifest.json` 放在 `hm_stability/sample/` 下。详细输入包由本地论文工作区生成。

在云端 Python 3.11+ 环境中安装 `requirements.txt`，运行 `python run_stability.py`。每个种子根据验证集 NDCG@10 选取模型轮次，然后计算测试集指标。运行结果输出到 `outputs/`。本仓库代码不包含 805 平台数据，也不包含用户原始 ID。
