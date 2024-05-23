### 项目复现说明
  本代码可直接复现FIGURE 4、FIGURE 6、section 3.2 中的Kernel准确率、Figure 12

### 运行环境
- Python 版本: 3.10.13
- 相关库版本：requirements.txt

### 相关文件说明
- `Display_images_of_the_eight_bushings`    # Information on visible light and infrared images of the eight bushings.
- `Data`  
  - `Raw_data`                              # 代码运行必要文件。套管的原始红外图像数据。
  - `Generated_data`                        # 代码运行非必要文件。代码运行过程中生成的相关数据。
  - `mlp_cpu_model.pth`                     # 代码运行必要文件。红外图像中，最大最小温度数值识别模型。
  - `mlp_model.pth`                         # 代码运行必要文件。红外图像中，最大最小温度数值识别模型。
- `subfunction`                             # 代码运行必要文件。子函数。
- `result`                                  # 代码运行非必要文件。所有结果存储位置。
- `main.py`                                 # 代码运行必要文件。主函数。

## 代码运行说明
```
所有结果复现：
  python main.py all

FIGURE 4 混淆矩阵结果复现运行代码：
  python main.py matrix

FIGURE 6 准确率复现运行代码：
  python main.py matrix

section 3.2 中的Kernel准确率复现运行代码：
  python main.py kernel

Figure 12 无监督算法组合结果复现运行代码：
  python main.py matrix
```

