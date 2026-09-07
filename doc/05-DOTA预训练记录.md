# DOTA 车辆检测预训练完成记录

> 日期：2026-09-07
> 结论：预训练完成，DOTA 验证集 mAP50=0.812；但在成都冬季影像上检出为 0，印证"冬季影像车辆不可用"结论，权重留待成都夏季影像/GE 截图微调后使用。

---

## 一、数据转换

- 输入：DOTA v1.0 影像（train 1411 + val 458 张）+ v1.5 标注（加密超集，8 坐标旋转框）
- 只保留 small-vehicle / large-vehicle / car，合并为单一 vehicle 类
- 1024×1024 切片（重叠 200），多边形裁剪 + minAreaRect 转旋转框
- 产物：**train 4260 切片 / val 1220 切片**（脚本 `scripts/04_dota_to_yolo_obb.py`）
- 类别分布确认：small-vehicle 5955、large-vehicle 1157（train 抽样 30 文件），车辆类占绝对主导

## 二、训练

- 模型：YOLO11n-OBB，imgsz 896，batch 4（8GB 显存安全值，初试 batch8@1024 触发 `bad allocation`）
- 进度：5 小时跑到 **epoch 27/80** 后人工停止（按此速度全程需约 15 小时，边际收益递减）
- 显存：常规 epoch 2.1G，高实例 batch 峰值 8.72G（顶格）
- 产物：`runs/obb/dota_vehicle_obb/weights/best.pt`（epoch 27）
- 日志：`runs_train_log.txt`

## 三、评估结果

**DOTA 验证集（1220 切片）**：mAP50 = **0.812**，mAP50-95 = **0.518**
- Precision 0.772 / Recall 0.755 / 57893 实例
- 单图推理 14.4ms（GPU）

**成都冬季影像（JL1KF01C 0.5m，2023-12）**：检出 **0** 辆（conf 0.25 与 0.05 均无）
- 原因：影像亮度均值 31/255，车辆与沥青路面对比度极低；DOTA 训练影像（gsd≈0.146m、光照好）与成都冬季影像域差异巨大
- 印证：问题在影像不在模型，与 `doc/04` 质量评估结论一致

## 四、结论与后续

1. 权重本身有效（DOTA 域 mAP50 0.81），**留待成都夏季影像 / GE 截图微调**后使用；
2. 夏季影像到位后：直接在该权重上微调（而非重新预训练）；
3. GE 成都截图（gsd 约 0.15-0.3m，光照正常）与 DOTA 训练域接近，预期微调后能获得可用检测器；
4. 车辆检测模块当前**卡在数据端**（缺可用的成都目标影像），不卡在模型端。

## 五、经验教训

- RTX 5060 8GB 训练 OBB@1024：batch 4 + imgsz 896 是稳定配置；
- Ultralytics OBB 训练避免自定义 project（默认 runs/obb，叠加会嵌套）；
- Windows 下训练日志重定向到文件避免控制台缓冲丢失；
- 高实例 batch 会瞬间拉满显存，遇 `bad allocation` 降 batch 而非 imgsz。
