# 论文附带代码检查与评估补齐（2026-11-08 / 2026-11-09）

## 结论

本文件夹包含实际可执行的关键 PyTorch 训练代码，可作为论文附带训练代码的主体。默认论文配置下未发现阻断性训练缺漏。模型、损失、优化器、梯度累积、数据增强、验证选模和检查点恢复均有实现，统一训练入口实际调用这些模块。

本次检查增加了最小 CPU 训练和恢复验收；它不代表重新完成了 30 次正式训练或重现论文成绩。原始模型权重、数据、结果记录和冻结核心源码均未修改。

## 代码定位

| 功能 | 文件 / 入口 |
|---|---|
| 五模型统一训练命令 | `scripts/train.py` |
| 五模型独立测试评估 | `scripts/evaluate.py`、`src/waste_reliability/evaluation.py` |
| 训练轮次、验证选模、best/last 保存、恢复和最终测试 | `src/waste_reliability/_portable_loop.py::train_run` |
| 模型选择、训练函数分派、路径适配和固定配置检查 | `src/waste_reliability/portable.py` |
| 通用训练与反向传播 | `src/waste_reliability/closed_set_baseline.py::train_epoch` |
| Dataset、训练增强、DataLoader、AdamW、warmup/cosine | `src/waste_reliability/closed_set_baseline.py` |
| PMAM | `src/waste_reliability/convnext_pmam.py` |
| WTConv | `src/waste_reliability/convnext_wtconv.py` |
| WTConv + PMAM | `src/waste_reliability/convnext_pmam_wtconv.py` |
| WaveTriNeXt 模型与 W1/W2/F3 引导 | `src/waste_reliability/convnext_wavelet_guided_trihead_fg01.py` |
| WaveTriNeXt 辅助损失、反向传播及评估 | `src/waste_reliability/wavelet_guided_trihead_fg01_training.py` |

真实训练步骤包括 `loss.backward()`、按累积窗口实际样本数归一化梯度、`clip_grad_norm_`、`optimizer.step()` 和 `scheduler.step()`。默认有效 batch 为 32、microbatch 为 8；使用 FP32，AMP/TF32 关闭。

WaveTriNeXt 的目标为主分类交叉熵 + 0.1 × 前景交叉熵 + 0.1 × 背景抑制损失。W1/W2/F3 引导路径保留梯度；验证指标和最终预测使用主分类头。检查点按验证集 Macro-F1 选择，同分时比较主头普通交叉熵，再优先较早轮次。

## 本次实际执行的检查

详细记录见 [`provenance/training_audit.json`](../provenance/training_audit.json)。

1. 五个模型均用官方 ImageNet 初始化构建，在 CPU 上处理 6 张 224×224 合成图片，执行真实反向传播和两次参数更新。该测试使用 microbatch=2、测试有效 batch=4，覆盖完整累积窗口和最后不足一个窗口的情况，不作为论文超参数运行。
2. 各模型骨干参数确实更新；骨干、分类头、PMAM、WTConv、TriHead 对应优化器组（按模型适用范围）均收到有限、非零梯度。WaveTriNeXt 总损失与三项加权损失一致。
3. 五组 TrashNet seed14 的原始 `best.pt` 均通过 `weights_only=True` 加载与严格 `state_dict` 匹配，随后前向输出正常。
4. 直接执行实际便携训练事务 `_portable_loop.train_run`：使用历史训练集中的 18 张图片，临时构建每类均有样本的 6/6/6 审计划分；运行 WaveTriNeXt 两轮，在第一轮保存后人为中断，再从 `last.pt` 恢复第二轮。模型、优化器、调度器和 RNG 恢复、best/last 保存、最终预测与指标导出、完成文件哈希均通过。

临时训练产物已清理。审计划分和分数不进入论文结果表。此前已经完成的 30 份正式权重 SHA256 核验记录保留在 `provenance/checkpoint_validation.json`。

## 独立评估入口（2026-11-09 已补齐）

2026-11-08 检查指出的独立评估缺口已经补齐。`scripts/evaluate.py` 加载
`pretrained/MODEL/DATASET/seed_N/best.pt`，按固定测试集读取真实图片并重新运行模型，
导出预测、指标、分类报告、混淆矩阵和按种子汇总的均值/样本标准差。
具体命令见 [EVALUATION.md](EVALUATION.md)。

评估会验证权重 SHA256、原实验身份、类别顺序、固定数据清单和测试图片摘要；
复用原始预处理与主分类头，并按 `sample_id` 对齐存档预测后报告差异。
完整测试和少量图片检查分别标记，新输出不会覆盖原归档或已有评估目录。

- `scripts/train.py --weights` 接收官方 ImageNet 初始化文件，不接收论文训练后的 `best.pt`。
- `scripts/summarize_results.py` 从已有 `test_predictions.csv` 重算表格，不运行模型推理。
- `scripts/verify_pretrained.py` 校验文件摘要与实验身份，不计算模型测试成绩。
- `scripts/evaluate.py` 直接使用论文权重执行新的测试集推理。

独立评估的实际推理和功能测试记录见
[`provenance/evaluation_validation.json`](../provenance/evaluation_validation.json)。

## 正式发布前的其他收尾

由论文作者填写论文题目、作者、正式引用信息和项目代码许可证。便携训练循环保留主训练/验证指标，但日志列少于历史 WaveTriNeXt 的完整辅助损失及区域质量诊断；如希望新训练报告与历史报告逐列一致，应补齐这些诊断输出，训练目标本身不需要改变。
