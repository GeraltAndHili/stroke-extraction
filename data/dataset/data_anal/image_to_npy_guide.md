# image 到项目数据格式的转换说明

这个仓库里实际有两层数据格式：

1. `dataset/RHSEDB/*.npz`
   这是更完整的原始样本格式，`SDNet` 和推理脚本会直接读它。
2. `dataset_forSegNet_ExtractNet_RHSEDB/*.npy`
   这是从上面的 `npz` 拆出来的 5 个数组文件，`SegNet` 和 `ExtractNet` 训练会直接读它。

## 1. 原始 `npz` 里每个字段的含义

- `name`: 字符本身，例如 `诲`
- `stroke_name`: 笔画名称字符串，例如 `点,横折提,撇,...`
- `stroke_label`: 每一笔的 24 类笔画标签，范围 `0..23`
- `reference_color_image`: `(3,256,256)`，参考笔画叠加后的彩色图
- `reference_single_image`: `(N,256,256)`，参考单笔二值图
- `reference_single_centroid`: `(N,2)`，参考单笔质心
- `target_image`: `(1,256,256)`，目标整字二值图
- `target_single_image`: `(N,256,256)`，目标单笔二值图

## 2. `SegNet/ExtractNet` 用的 5 个 `npy`

对于样本编号 `id`，对应关系是：

- `id_kaiti_color.npy` <- `reference_color_image`
- `id_single.npy` <- `reference_single_image`
- `id_style_single.npy` <- `target_single_image`
- `id_seg.npy` <- `stroke_label`
- `id_style.npy` <- 8 通道数组，当前仓库实际只依赖第 0 通道的整字二值图

注意：

- 当前仓库在训练时会用 `id_style_single.npy + id_seg.npy` 重新聚合 7 类分组标签。
- 因此 `id_style.npy` 的后 7 个通道在当前实现里并不是关键输入。

## 3. 你的 `dataset/image` 这种标注能直接提供什么

你现在的图片结构是：

- `dataset/image/0_1.png`
  整字图，透明背景，字符区域可转成 `target_image`
- `dataset/image/strokes/0_1_stroke_*.png`
  每张图保留整字灰色背景，并把当前笔画涂成其他颜色

这类标注可以稳定提取出：

- `target_image`
- `target_single_image`
- 如果没有单独参考字，也可以先把 `reference_single_image` 临时设成和 `target_single_image` 一样
- `reference_single_centroid`
- `reference_color_image`

## 4. 唯一不能只靠颜色自动恢复的字段

`stroke_label` 不能只靠图片颜色自动恢复。

原因是模型里要求的是 24 类笔画类别编号，不是“第几笔”也不是“像素颜色值”。  
所以你至少还需要下面三种信息中的一种：

1. 每一笔对应的 24 类标签序列，例如 `0,12,3,1,20,7`
2. 每一笔的笔画名称，再由你自己映射成 24 类编号
3. 一份颜色到 24 类标签的固定映射表

如果没有这个标签，脚本仍然可以生成结构正确的文件，但标签只能用占位值，不能作为真正的模型训练标签。

## 5. 已经新增的转换脚本

脚本路径：

- [convert_image_folder_to_project_dataset.py](/f:/cha_extraction/StrokeExtraction/convert_image_folder_to_project_dataset.py)

示例：

```powershell
python convert_image_folder_to_project_dataset.py `
  --sample-id 0_1 `
  --char-name 永 `
  --stroke-labels 0,1,2,3,4,5 `
  --stroke-names 点,横,撇,捺,竖,钩
```

输出位置默认在：

- `dataset/data_anal/converted_from_image/0_1/`

里面会同时生成：

- `0_1.npz`
- `0_1_kaiti_color.npy`
- `0_1_single.npy`
- `0_1_style_single.npy`
- `0_1_style.npy`
- `0_1_seg.npy`
- `meta.json`

## 6. 用你的图片生成“可训练/可推理”数据时的建议

- 如果你只是想先跑通数据结构：
  可以把 `reference_single_image = target_single_image`
- 如果你想尽量贴近原论文训练设定：
  最好再准备同一个字的“参考分解笔画”，不要直接复用目标笔画
- 如果你要用于 `SegNet/ExtractNet` 训练：
  先确保 `stroke_label` 是真实 24 类标签
- 如果你要用于仓库里的推理脚本：
  也同样需要真实 `stroke_label`，因为推理阶段会用它把单笔聚合成 7 类结构信息
