运行环境：python==3.10.13
库的版本：requirements.txt


相关文件说明：                             
-Display_images_of_the_eight_bushings      #   Information on visible light and infrared images of the eight bushings         
-Data                   
  -Raw_data             # 代码运行必要文件。套管的原始红外图像数据
  -Generated_data       # 代码运行非必要文件。代码运行过程中生成的相关数据
  -mlp_cpu_model.pth    # 代码运行必要文件。红外图像中，最大最小温度数值识别模型
  -mlp_model.pth        # 代码运行必要文件。红外图像中，最大最小温度数值识别模型
-subfunction            # 代码运行必要文件。子函数
-result                 # 代码运行非必要文件。所有结果存储位置
main.py                 # 代码运行必要文件。主函数

代码运行及结果说明：
所有结果展示运行代码：
  python main.py all
  代码生成****


section 3.2 中Kernel准确率结果复现运行代码：
  python main.py kernel
  代码生成****
  
图12。混淆矩阵结果复现运行代码：
  python main.py matrix
   代码生成****
