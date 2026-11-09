# Paper model checkpoints / 论文模型权重

当前已导入并核验 **30 / 30** 份最佳权重。每个模型计划包含两个数据集 × 种子 14、42、70，共六份。

| 配置 | 已核验 / 计划 | 当前状态 |
|---|---:|---|
| baseline | 6 / 6 | verified |
| pmam | 6 / 6 | verified |
| wtconv | 6 / 6 | verified |
| wtconv_pmam | 6 / 6 | verified |
| wavetrinext | 6 / 6 | verified |

## 文件与校验

```text
pretrained/MODEL/DATASET/seed_N/best.pt
```

`manifest.json` 保存每份权重的期望 SHA256、原始实验对应关系和实际导入状态。只有与对应 `results/MODEL/DATASET/seed_N/complete.json` 的 `artifacts_sha256.best.pt` 一致，才记为 `verified`。权重本身保持原始字节，未转换格式。

在仓库根运行：

```bash
python scripts/verify_pretrained.py
# 如需要求全部 30 份均已到位：
python scripts/verify_pretrained.py --require-all
```

状态含义：`verified` 已导入且哈希通过；`connection_unavailable` 当前无法连接云主机，尚不能判断云端文件是否存在；`not_found_on_checked_host` 已连接检查但未找到；`available_for_download` 已定位待下载；`transfer_interrupted` 传输中断；`remote_record_mismatch` 或 `download_hash_mismatch` 表示来源或内容不匹配，文件不计入有效模型。

这里保存的是验证集选出的论文 `best.pt`，用于推理或评估。训练入口 `scripts/train.py --weights` 接收的是官方 ImageNet 初始化文件，不能直接把这里的 `best.pt` 当作该参数。`last.pt` 用于接续训练，不在本次导入范围内。

这些大文件已配置 Git LFS；私有云主机地址与下载记录保留在发布文件夹外。目前没有上传到 GitHub。
