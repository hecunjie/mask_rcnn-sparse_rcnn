import mmcv
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
import os
import torch
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer
import numpy as np


pred_score_thr = 0.3

# 模型配置
mask_config_file = 'work_dirs/mask_rcnn_voc/mask-rcnn_r50_fpn_1x_voc.py'
mask_checkpoint_file = 'work_dirs/mask_rcnn_voc/best_coco_bbox_mAP_epoch_43.pth'
sparse_config_file = 'work_dirs/sparse_rcnn_voc/sparse-rcnn_r50_fpn_1x_coco.py'
sparse_checkpoint_file = 'work_dirs/sparse_rcnn_voc/best_coco_bbox_mAP_epoch_98.pth'
device = 'cuda:0'  # 或 'cpu'

# 初始化模型
mask_model = init_detector(mask_config_file, mask_checkpoint_file, device=device)
sparse_model = init_detector(sparse_config_file, sparse_checkpoint_file, device=device)

# 图像路径
# data_root = '/home/hcj_24210980089/mmdetection/data/coco/VOC2012/JPEGImages'
# imgs = [
#     '2007_000027.jpg',
#     '2007_000032.jpg',
#     '2007_000033.jpg',
#     '2007_000039.jpg',
# ]
data_root = '/home/hcj_24210980089/mmdetection/work_dirs/extra_photos'
imgs = [
    "bird.jpg",
    "dog.jpg",
    "human.jpg",
    "plane.jpg",
]
image_paths = [os.path.join(data_root, img) for img in imgs]

# 输出目录
output_dir = 'work_dirs/output'
os.makedirs(output_dir, exist_ok=True)

# 初始化可视化器
visualizer = DetLocalVisualizer(
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir=output_dir
)
visualizer.dataset_meta = dict(classes=mask_model.dataset_meta['classes'])

# 可视化函数
def visualize_proposals_and_results(img_path, proposals, mask_result, output_path,pred_score_thr=0.3):
    img = mmcv.imread(img_path)
    img_proposal = img.copy()
    img_result = img.copy()

    # 可视化 proposal box
    if proposals is not None:
        for i in range(min(10, len(proposals.bboxes))):
            bbox = proposals.bboxes[i].cpu().numpy().astype(int)
            score = proposals.scores[i].cpu().numpy()
            if score > 0.9:
                mmcv.imshow_bboxes(
                    img_proposal,
                    bbox[None, :],  # 单框输入
                    colors=(0, 255, 0),  # 绿色
                    thickness=1,
                    show=False
                )

    # 可视化 Mask R-CNN 最终结果
    mask_data_sample = DetDataSample()
    mask_data_sample.pred_instances = mask_result.pred_instances
    mask_data_sample.set_metainfo(dict(
        img_path=img_path,
        ori_shape=img.shape[:2],
        img_shape=img.shape[:2],
        scale_factor=np.array([1.0, 1.0, 1.0, 1.0])
    ))
    visualizer.add_datasample(
        'mask_rcnn',
        img_result,
        mask_data_sample,
        draw_gt=False,
        show=False,
        pred_score_thr=pred_score_thr
        # with_labels=True
    )
    img_result = visualizer.get_image()

    # 并排显示
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title('Proposal Boxes')
    plt.imshow(img_proposal[:, :, ::-1])
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Final Prediction (Mask R-CNN)')
    plt.imshow(img_result[:, :, ::-1])
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# 可视化 Sparse R-CNN 结果
def visualize_sparse_rcnn_results(img_path, sparse_result, output_path,pred_score_thr=0.3):
    img = mmcv.imread(img_path)
    sparse_data_sample = DetDataSample()
    sparse_data_sample.pred_instances = sparse_result.pred_instances
    sparse_data_sample.set_metainfo(dict(
        img_path=img_path,
        ori_shape=img.shape[:2],
        img_shape=img.shape[:2],
        scale_factor=np.array([1.0, 1.0, 1.0, 1.0])
    ))
    visualizer.add_datasample(
        'sparse_rcnn',
        img,
        sparse_data_sample,
        draw_gt=False,
        show=False,
        pred_score_thr=pred_score_thr
        # with_labels=True
    )
    img_result = visualizer.get_image()
    mmcv.imwrite(img_result, output_path)

# 对比 Mask R-CNN 和 Sparse R-CNN
def visualize_comparison(img_path, mask_result, sparse_result, output_path,pred_score_thr=0.3):
    img = mmcv.imread(img_path)
    img_mask = img.copy()
    img_sparse = img.copy()

    # Mask R-CNN 结果
    mask_data_sample = DetDataSample()
    mask_data_sample.pred_instances = mask_result.pred_instances
    mask_data_sample.set_metainfo(dict(
        img_path=img_path,
        ori_shape=img.shape[:2],
        img_shape=img.shape[:2],
        scale_factor=np.array([1.0, 1.0, 1.0, 1.0])
    ))
    visualizer.add_datasample(
        'mask_rcnn',
        img_mask,
        mask_data_sample,
        draw_gt=False,
        show=False,
        pred_score_thr=pred_score_thr
        # with_labels=True
    )
    img_mask = visualizer.get_image()

    # Sparse R-CNN 结果
    sparse_data_sample = DetDataSample()
    sparse_data_sample.pred_instances = sparse_result.pred_instances
    sparse_data_sample.set_metainfo(dict(
        img_path=img_path,
        ori_shape=img.shape[:2],
        img_shape=img.shape[:2],
        scale_factor=np.array([1.0, 1.0, 1.0, 1.0])
    ))
    visualizer.add_datasample(
        'sparse_rcnn',
        img_sparse,
        sparse_data_sample,
        draw_gt=False,
        show=False,
        pred_score_thr=pred_score_thr
        # with_labels=True
    )
    img_sparse = visualizer.get_image()

    # 并排显示
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title('Mask R-CNN')
    plt.imshow(img_mask[:, :, ::-1])
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Sparse R-CNN')
    plt.imshow(img_sparse[:, :, ::-1])
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# 主循环
for img_path in image_paths:
    # Mask R-CNN 推理
    mask_result = inference_detector(mask_model, img_path)

    # 获取 RPN proposals
    proposals = None
    if hasattr(mask_model, 'rpn_head'):
        # 手动预处理图像
        img = mmcv.imread(img_path)
        img = mmcv.imrescale(img, scale=(1333, 800))
        img = mmcv.imnormalize(img,
                               mean=np.array([123.675, 116.28, 103.53]),
                               std=np.array([58.395, 57.12, 57.375]),
                               to_rgb=True)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        # 提取特征
        feats = mask_model.extract_feat(img_tensor)

        # 构造 batch_data_samples
        batch_data_samples = [
            DetDataSample(
                metainfo=dict(
                    ori_shape=mmcv.imread(img_path).shape[:2],
                    img_shape=img.shape[:2],
                    scale_factor=np.array([1.0, 1.0, 1.0, 1.0]),
                    pad_shape=img.shape[:2]
                )
            )
        ]

        # 获取 proposals
        proposals = mask_model.rpn_head.predict(feats, batch_data_samples, rescale=False)[0]


    # 可视化 Mask R-CNN proposals 和最终结果
    output_path = os.path.join(output_dir, f'mask_rcnn_{os.path.basename(img_path)}')
    visualize_proposals_and_results(img_path, proposals, mask_result, output_path, pred_score_thr=pred_score_thr)

    # Sparse R-CNN 推理和可视化
    sparse_result = inference_detector(sparse_model, img_path)
    output_path = os.path.join(output_dir, f'sparse_rcnn_{os.path.basename(img_path)}')
    visualize_sparse_rcnn_results(img_path, sparse_result, output_path,pred_score_thr=pred_score_thr)

    # 对比可视化
    output_path = os.path.join(output_dir, f'comparison_{os.path.basename(img_path)}')
    visualize_comparison(img_path, mask_result, sparse_result, output_path,pred_score_thr=pred_score_thr)
