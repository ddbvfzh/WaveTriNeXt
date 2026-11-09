# Datasets / 数据集

本目录包含论文五组消融实验使用的 **TrashNet 六类** 和 **RealWaste 九类** 数据，不包含旧项目的五类跨域实验或 TACO 数据。

## 内容与固定划分

| Dataset | 原始图片 | 论文纳入图片 | Train | Validation | Test | 原始图片体积 |
|---|---:|---:|---:|---:|---:|---:|
| TrashNet | 2,527 | 2,521 | 1,765 | 379 | 377 | 43,282,402 bytes |
| RealWaste | 4,752 | 4,752 | 3,327 | 713 | 712 | 688,308,374 bytes |
| Total | 7,279 | 7,273 | 5,092 | 1,092 | 1,089 | 731,590,776 bytes |

所有 JPEG 按原字节复制；仅调整目录位置，没有重编码、裁剪或修改标签。TrashNet 为作者发布的 `dataset-resized` 版本（512 × 384），不是原始全分辨率摄影档案。完整数据目录约 698 MiB；此处的体积是图片字节总数，未包含 CSV/JSON 文档。

```text
datasets/
├── dataset_inventory.json
├── trashnet/
│   ├── images/{cardboard,glass,metal,paper,plastic,trash}/
│   ├── manifest.csv
│   ├── splits/{train,val,test}.csv
│   ├── raw_image_inventory.csv
│   ├── class_counts.csv
│   ├── excluded.csv
│   ├── duplicates.csv
│   └── provenance/
└── realwaste/
    ├── images/{cardboard,food_organics,glass,metal,miscellaneous_trash,
    │          paper,plastic,textile_trash,vegetation}/
    └── ... (same metadata structure)
```

训练时读取 `manifest.csv`，不要使用整个 `images/` 自动重新划分。TrashNet 中 3 组共 6 张图片具有相同内容、冲突标签，原实验将这些图片全部排除；本资料包仍保留原图，并在 `excluded.csv` 中记录。另有一个 macOS 归档元数据项不作为图片复制，`excluded.csv` 用 `source_archive/trashnet.zip!/` 前缀标记它在原压缩包中的位置。RealWaste 没有排除图片。

划分 seed 为 42，目标比例为 70% / 15% / 15%。文件和 RGB 像素摘要用于精确重复检测，pHash 汉明距离不超过 4 的近重复图片按连通组放入同一划分；因此最终比例略有偏差。实验训练 seed 为 14、42、70，三个训练 seed 使用同一数据划分。

已核对 `baseline`、`pmam`、`wtconv`、`wtconv_pmam`、`wavetrinext` 五组原始清单：除图片绝对路径外，所有列和样本分配一致。核对记录及原始清单 SHA256 在各数据集 `provenance/split_equivalence.json` 中。原有绝对路径已转为相对于仓库根目录的 POSIX 路径，保留 `sample_id`、`dataset`、`class_name`、`label`、`group_id`、`sha256`、`pixel_sha256`、`phash` 和 `split`。

`raw_image_inventory.csv` 包含全部原图、每张图片 SHA256、原始相对路径和字节数。`provenance/source_audit_summary.json` 为历史审计摘要，移除了机器专属根路径；其中历史文件摘要不是当前可移植 CSV 的摘要，当前摘要记录在 `dataset_inventory.json`。

## 验证

在仓库根目录运行，Python 标准库即可：

```bash
python scripts/verify_datasets.py
```

默认核验全部 7,279 张图片的 SHA256、固定划分、标签、重复组不跨划分、分割 CSV 一致性和图片数量。可选 `--skip-image-hashes` 仅跳过图片内容摘要，用于快速检查目录结构和文件大小；完整验收应使用默认命令。

## 来源、引用和许可

**TrashNet**：Gary Thung 与 Mindy Yang 创建；来源为 [作者仓库](https://github.com/garythung/trashnet) 中的 `data/dataset-resized.zip`。作者仓库标记为 MIT，原文保存在 `trashnet/provenance/UPSTREAM_LICENSE.txt`。引用作者仓库及其 [2016 年课程论文](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf)。

**RealWaste**：Sam Single、Saeid Iranmanesh、Raad Raad 创建；本地档案从 [UCI RealWaste](https://www.archive.ics.uci.edu/dataset/908/realwaste) 下载。数据集 DOI：[10.24432/C5SS4G](https://doi.org/10.24432/C5SS4G)；论文：[RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning](https://doi.org/10.3390/info14120633)。

RealWaste 的来源许可声明存在差异：UCI 页面标记为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，而[原作者仓库](https://github.com/sam-single/realwaste)及下载档案中的 README 标记为 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。下载档案 README 原文保存在 `realwaste/provenance/UPSTREAM_README.md`。本整理工作保留这两份来源声明，没有重新授权数据；公开分发图片时需保留作者署名、引用及适用的数据许可，代码许可证不覆盖数据集。来源于 2026-11-08 核对。

原始下载 URL、下载日期、ZIP 字节数和 SHA256 保存在各自 `provenance/source_download.json`。本目录包含本地完整副本；是否将大体积图片上传 GitHub，以及是否通过 Git LFS 或独立数据存储分发，见仓库发布说明。
