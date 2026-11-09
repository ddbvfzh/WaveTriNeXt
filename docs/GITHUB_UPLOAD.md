# GitHub 私有仓库与大文件

仓库为 [ddbvfzh/WaveTriNeXt](https://github.com/ddbvfzh/WaveTriNeXt)，默认分支 `main`，当前设为**私有**。访问和克隆需要获得授权的 GitHub 账号。上传范围为本文件夹，父目录的私有云端位置说明不在仓库中。

代码、配置、固定数据划分、30 次实验结果及验证记录存储在普通 Git 中；两套数据的 **7,279 张图片和 30 份 best.pt** 使用 Git LFS。模型位置为 `pretrained/模型名/数据集/seed_N/best.pt`，清单见 `pretrained/manifest.json`。

## 1. 检查整理内容

在 `WaveTriNeXt` 文件夹内执行：

```bash
python scripts/verify_datasets.py
python scripts/summarize_results.py
python scripts/verify_pretrained.py
```

数据来源与许可说明在 [datasets/README.md](../datasets/README.md)，项目代码许可说明在 [LICENSE_POLICY.md](../LICENSE_POLICY.md)。论文题目、作者、正式 DOI 尚未提供，本整理包没有编造这些信息。

## 2. 后续更新

本地独立仓库已初始化并配置 Git LFS。后续修改、检查文件后执行：

```bash
git status --short
git add .
git commit -m "Update WaveTriNeXt research code"
git push origin main
```

`.gitattributes` 已将两套数据图片及 `pretrained/**/*.pt` 指定为 LFS 文件。共有 **7,309 个 LFS 文件路径、7,306 个唯一内容对象**；TrashNet 中 3 对图片内容相同，LFS 去重存储，所以默认 `git lfs ls-files` 显示 7,306 项，不代表文件缺失。数据图片约 698 MiB，权重约 3.53 GB，LFS 实际内容总计约 4.26 GB。Git LFS 在 Git 仓库中记录指针，实际内容存入 LFS 存储，克隆后需要执行 `git lfs pull`。详细规则见 [GitHub 官方 Git LFS 说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

每次更新大文件都会产生新的 LFS 内容版本。存储与下载流量按仓库所有者的额度计算，规则见 [GitHub LFS 计量说明](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)。本次上传不修改账户的预算或付费设置。

`.gitignore` 允许提交 `pretrained/**/best.pt`，仍忽略其他位置的模型权重、`outputs/`、缓存和下载中间文件。只有核验完成的文件才应以 `best.pt` 名称进入该目录；下载和 SHA256 核验尚未完成时，不要把临时文件当作最终权重提交。

`.gitattributes` 同时关闭自动文本换行转换，以保留实验记录和冻结源码的 SHA256。不要用编辑器批量转换 `results/` 与 `provenance/` 的换行符。

## 3. 下载已发布仓库

```bash
git clone https://github.com/ddbvfzh/WaveTriNeXt.git
cd WaveTriNeXt
git lfs install --local
git lfs pull
python scripts/verify_datasets.py
python scripts/summarize_results.py
python scripts/verify_pretrained.py --require-all
```

训练说明见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)，独立评估见 [EVALUATION.md](EVALUATION.md)。仓库当前用于私有整理；向读者提供论文代码链接前，需要由所有者决定公开发布或授予访问权限。
