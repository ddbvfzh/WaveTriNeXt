# 上传 GitHub

本文件夹是待发布的独立仓库内容。两个数据集的真实图片已放入 `datasets/`，不是下载链接或占位文件。模型权重的实际导入情况以 `pretrained/manifest.json` 为准；可用的最佳权重统一位于 `pretrained/模型名/数据集/seed_N/best.pt`，缺失项保留在清单中。

## 1. 检查整理内容

在 `WaveTriNeXt` 文件夹内执行：

```bash
python scripts/verify_datasets.py
python scripts/summarize_results.py
python scripts/verify_pretrained.py
```

数据来源与许可说明在 [datasets/README.md](../datasets/README.md)，项目代码许可说明在 [LICENSE_POLICY.md](../LICENSE_POLICY.md)。论文题目、作者、正式 DOI 尚未提供，本整理包没有编造这些信息。

## 2. 初始化并提交

先安装 Git 和 Git LFS，在 GitHub 建立自己的空仓库，再在本文件夹执行以下命令。命令中的仓库地址需要换成自己的地址。

```bash
git init -b main
git lfs install --local
git add .
git lfs ls-files
git status --short
git commit -m "Add WaveTriNeXt code, datasets, results and available checkpoints"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

`.gitattributes` 已将两套数据图片及 `pretrained/**/*.pt` 指定为 LFS 文件。`git lfs ls-files` 应列出 **7,279 张图片，加上清单中实际导入的 best.pt 数量**。数据图片约 698 MiB，权重占用另计，需要账户有相应的 LFS 存储和传输额度。Git LFS 在 Git 仓库中记录指针，实际内容存入 LFS 存储；详细规则见 [GitHub 官方 Git LFS 说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

`.gitignore` 允许提交 `pretrained/**/best.pt`，仍忽略其他位置的模型权重、`outputs/`、缓存和下载中间文件。只有核验完成的文件才应以 `best.pt` 名称进入该目录；下载和 SHA256 核验尚未完成时，不要把临时文件当作最终权重提交。

`.gitattributes` 同时关闭自动文本换行转换，以保留实验记录和冻结源码的 SHA256。不要用编辑器批量转换 `results/` 与 `provenance/` 的换行符。

## 3. 下载已发布仓库

```bash
git clone https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
git lfs install --local
git lfs pull
python scripts/verify_datasets.py
python scripts/summarize_results.py
python scripts/verify_pretrained.py
```

整理过程中没有创建 GitHub 仓库、提交或推送。仅上传 `WaveTriNeXt/` 内部的内容；其父目录中的私有云端位置说明留在本地。
