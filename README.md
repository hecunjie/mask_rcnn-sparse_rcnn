# Mask R-CNN and Sparse R-CNN on VOC Dataset

## 项目概述

本项目基于 [MMDetection](https://github.com/open-mmlab/mmdetection) 框架，在 PASCAL VOC 数据集上训练和测试 Mask R-CNN 和 Sparse R-CNN 模型，用于目标检测和实例分割任务。项目涵盖了模型训练、性能评估、测试集图像可视化以及对额外图像的预测展示。

### 主要内容

- **模型**：Mask R-CNN 和 Sparse R-CNN，均采用 ResNet-50 和 FPN 架构。
- **数据集**：PASCAL VOC 2007 和 VOC 2012，包含 20 个物体类别。
- **可视化**：展示测试集图像的 proposal box、最终预测结果以及两个模型的对比。
- **额外图像**：对非 VOC 数据集中的图像进行预测，验证模型的泛化能力。

## 环境配置

### 依赖安装

1. 安装 Anaconda 并创建新环境：
   ```bash
   conda create -n mmdet python=3.8
   conda activate mmdet
   ```
2. 安装 PyTorch 和 torchvision（根据你的 CUDA 版本调整）：
   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   ```
3. 安装 MMDetection：
   ```bash
   pip install -U openmim
   mim install mmengine
   mim install "mmcv>=2.0.0"
   pip install mmdet
   ```

### 验证安装

运行以下命令确认安装成功：
```bash
python -c "import mmdet; print(mmdet.__version__)"
```

## 数据集准备

1. 下载 PASCAL VOC 2007 和 VOC 2012 数据集，并解压至 `./data/VOCdevkit`。
2. 确保目录结构如下：
   ```
   VOCdevkit/
   ├── VOC2007/
   │   ├── Annotations/
   │   ├── ImageSets/
   │   ├── JPEGImages/
   ├── VOC2012/
   │   ├── Annotations/
   │   ├── ImageSets/
   │   ├── JPEGImages/
   ```
3. 使用 MMDetection 提供的脚本生成 COCO 格式的标注文件。

## 模型训练

### Mask R-CNN

1. 使用配置文件 `work_dirs/mask_rcnn_voc/mask-rcnn_r50_fpn_1x_voc.py`。
2. 运行训练命令：
   ```bash
   python tools/train.py work_dirs/mask_rcnn_voc/mask-rcnn_r50_fpn_1x_voc.py --work-dir work_dirs/mask_rcnn_voc
   ```

### Sparse R-CNN

1. 使用配置文件 `work_dirs/sparse_rcnn_voc/sparse-rcnn_r50_fpn_1x_coco.py`。
2. 运行训练命令：
   ```bash
   python tools/train.py work_dirs/sparse_rcnn_voc/sparse-rcnn_r50_fpn_1x_coco.py --work-dir work_dirs/sparse_rcnn_voc
   ```

### 监控训练

使用 TensorBoard 查看训练过程：
```bash
tensorboard --logdir work_dirs/mask_rcnn_voc/tensorboard_logs
tensorboard --logdir work_dirs/sparse_rcnn_voc/tensorboard_logs
```

## 模型测试

### 测试集评估

1. Mask R-CNN：
   ```bash
   python tools/test.py work_dirs/mask_rcnn_voc/mask-rcnn_r50_fpn_1x_voc.py work_dirs/mask_rcnn_voc/best_coco_bbox_mAP_epoch_43.pth --eval mAP
   ```
2. Sparse R-CNN：
   ```bash
   python tools/test.py work_dirs/sparse_rcnn_voc/sparse-rcnn_r50_fpn_1x_coco.py work_dirs/sparse_rcnn_voc/best_coco_bbox_mAP_epoch_98.pth --eval mAP
   ```

### 可视化

使用 `test.py` 脚本生成可视化结果：
```bash
python test.py
```
结果保存在 `work_dirs/output` 目录下。

## 实验结果

### 性能比较

| 模型          | mAP@0.5 (%) | mAP@0.75 (%) |
|---------------|-------------|--------------|
| Mask R-CNN    | 78.5        | 54.2         |
| Sparse R-CNN  | 79.8        | 55.9         |



## 模型权重

- [Mask R-CNN 最佳权重](https://drive.google.com/file/d/1Uq-OhWE6VT16FJhMWcNUFqWLG6peqKJT/view?usp=sharing)
- [Sparse R-CNN 最佳权重](https://drive.google.com/file/d/1Uq-OhWE6VT16FJhMWcNUFqWLG6peqKJT/view?usp=sharing)

## 引用

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [Sparse R-CNN Paper](https://arxiv.org/abs/2011.12450)
