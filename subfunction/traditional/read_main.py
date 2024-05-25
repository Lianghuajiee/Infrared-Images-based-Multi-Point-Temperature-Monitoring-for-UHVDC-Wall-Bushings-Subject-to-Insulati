import numpy as np
import cv2 as cv
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import os
import sys
import copy
import matplotlib.image as mpimg
import time
import math
from PIL import Image as IMG
import json 
from subfunction.traditional.Denoising_model import smooth,NLM
from subfunction.traditional.Anomaly_detection import Model_selection , two_to_one
from subfunction.traditional.take_your_need import cut_image_conter,find_line_take_cannula_integration ,find_line_take_cannula ,piont_take_cannula
from tqdm import tqdm 
np.set_printoptions(threshold=1000000)

def Polygon_extraction(image,Outcome_site):
    Outcome_site = np.array([Outcome_site])
    # 和原始图像一样大小的0矩阵，作为mask
    mask = np.zeros(image.shape[:2], np.uint8)
    # 在mask上将多边形区域填充为白色
    cv.polylines(mask, Outcome_site, 1, 255,thickness=1)    # 描绘边缘
    cv.fillPoly(mask, Outcome_site, 255)    # 填充
    # 逐位与，得到裁剪后图像，此时是黑色背景
    dst_back = cv.bitwise_and(image, image, mask=mask)

     # 添加白色背景
    bg = np.ones_like(image, np.uint8) * 255
    cv.bitwise_not(bg, bg, mask=mask)  # bg的多边形区域为0，背景区域为255
    dst_white = bg + dst_back
    # cv.imshow("dst_white.jpg", dst_white)
    return dst_back, dst_white
def get_Preconditioning_data(model,path):

    with open('Preconditioning_data/'+model+'_'+path[:-4]+'.json', 'r') as json_file:
        loaded_data = json.load(json_file)
    return loaded_data        
def Convex_Hull (points):  #凸包顶点
    hull = ConvexHull(points)

    return points[hull.vertices]
 
def get_regression (max_tmp,min_tmp):   #获取回归线
    img_max = 255
    img_min = 0
    k = (max_tmp - min_tmp)/(img_max - img_min)
    b = min_tmp - k * img_min
    return k , b
    
def get_tmp (k , b , pixel):   #获取温度
    return round(pixel * k + b, 2)
    
def Segmented_analysis (piont,x_size,y_size):   

    piont_up_leng=np.unique(np.array(piont[:2]),axis=0)
    piont_donw_leng=np.unique(np.array(piont[2:]),axis=0)
    xpiont1 = int((piont_donw_leng[0,0]-piont_up_leng[0,0])/y_size)
    ypiont1 = int((piont_donw_leng[0,1]-piont_up_leng[0,1])/y_size)
    xpiont2 = int((piont_donw_leng[1,0]-piont_up_leng[1,0])/y_size)
    ypiont2 = int((piont_donw_leng[1,1]-piont_up_leng[1,1])/y_size)
    piont_Terminal  = []
    piont_Terminal.append(piont_up_leng)
    for i in range(1,y_size):
        piont_Terminal.append([[piont_up_leng[0,0]+i*xpiont1,piont_up_leng[0,1]+i*ypiont1],[piont_up_leng[1,0]+i*xpiont2,piont_up_leng[1,1]+i*ypiont2]])
    piont_Terminal.append(piont_donw_leng) 
    piont_Terminal = np.array(piont_Terminal)    
    
    piont_out = []
    
    for piont_list in piont_Terminal:
        x = (piont_list[1,0]-piont_list[0,0])/x_size
        y = (piont_list[1,1]-piont_list[0,1])/x_size
        qq = []

        qq.append(piont_list[0])
        for i in range(1,x_size):
            qq.append([int(piont_list[0,0]+x*i),int(piont_list[0,1] + y*i)])
        qq.append(piont_list[1])
        piont_out.append(qq)
    piont_out = np.array(piont_out)    
   
    return piont_out
    
def small_area_divide(image_gray,piont_up_mid_down,k,b,x_number,y_number):
    
    Small_area_temp_piont=[]
    Small_area_piont=[]
    Small_area_temp=[]
    Small_area_max_temp = []
    for i in range(x_number):
        for o in range(y_number):
            mid_area=[]
            for j in range(2):
                mid_area.append(piont_up_mid_down[0+o,i+j])
                mid_area.append(piont_up_mid_down[1+o,i+j])
            mid_area = np.array(mid_area)
            Small_area_temp_piont.append(mid_area)
            Small_area_piont.append(np.average(mid_area, axis=0))
            
    for itme in Small_area_temp_piont :             #可优化，减轻算力
        local= []   
        itme= Convex_Hull(itme)  
        dst_gray_small_area, _=Polygon_extraction(image_gray,itme)
        nonzero_coords = np.transpose(np.nonzero(dst_gray_small_area))     # 获取非零像素的坐标
        for coord in nonzero_coords:
            local.append(image_gray[coord[0], coord[1]])
        Small_area_temp.append(get_tmp(k , b , np.nanmean(local)))
        try:
            Small_area_max_temp.append(get_tmp(k , b ,max(local)))
        except:
            Small_area_max_temp.append(999)
    return Small_area_temp,Small_area_max_temp,Small_area_piont

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
 
        self.hidden1 = nn.Linear(8 * 12, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 13)
    
    def forward(self, x):

        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x
def model_predict(path):

    move_dir = ['0','1','2','3','4','5','6','7','8','9','.','-',' ']
    max_tem_size=[8,12]             #数字高12，宽8
    max_tem_piont = [37,7]
    min_tem_piont = [37,21]
    
    img = cv.imread(path,0) 
    img_max=img[max_tem_piont[1] : max_tem_piont[1]+max_tem_size[1] , max_tem_piont[0]:max_tem_piont[0]+max_tem_size[0]*5]
    img_min=img[min_tem_piont[1] : min_tem_piont[1]+max_tem_size[1] , min_tem_piont[0]:min_tem_piont[0]+max_tem_size[0]*5]
    max_data = ''
    min_data = ''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP()
    model.to(device)
    if device== "cuda" :
        model_name = 'mlp_model.pth'
    else:
        model_name = 'mlp_cpu_model.pth'
    model.load_state_dict(torch.load('../'+model_name))
    model.eval()
    for i in range(5):
        resized_image=img_max[:,max_tem_size[0]*i:max_tem_size[0]*(i+1)]
        resized_image = resized_image.reshape(-1)   
            
        img_tensor = torch.tensor(resized_image, dtype=torch.float32).to(device)
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor.float())
        max_data += move_dir[outputs.argmax(1)]
        
        resized_image=img_min[:,max_tem_size[0]*i:max_tem_size[0]*(i+1)]
        resized_image = resized_image.reshape(-1)   
            
        img_tensor = torch.tensor(resized_image, dtype=torch.float32).to(device)
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor.float())
        min_data += move_dir[outputs.argmax(1)]    
    max_data = eval(max_data)
    min_data = eval(min_data)
    return max_data , min_data             
     
if __name__ == "__main__":
    # path_list = os.listdir('../local_data_2_reserve/')
    path_list = ['local_10_0 (4).jpg','local_12_0 (2).jpg']#,'local_02_0 (4).jpg']        # 原图
    
    temp_try_list = {"tr_pro_diff": {}, "loca_small_mean_diff": {}, "loca_small_max_diff": {}}
    temp_try_list["tr_pro_diff"] = {}
    temp_try_list["loca_small_mean_diff"] = {}
    temp_try_list["loca_small_max_diff"] = {}
    model_name ='O_Z_normal_H/'
    
    piont_take_cannula.save_argument_to_txt()     
    find_line_take_cannula.save_argument_to_txt()    
    cut_list = [[2,5]]#,[2,10],[1,10],[1,5],[1,3],[2,3]]    
    for path in tqdm(path_list): 
        max_tmp , min_tmp = model_predict('../local_data_2_reserve/'+path)
        k , b =  get_regression(max_tmp,min_tmp)
        image = cv.imread('../local_data_2_reserve/'+path)                     # 图片输入
        img0 = cv.cvtColor(image,cv.COLOR_BGR2GRAY)         # 将彩色图片转换为灰度图片
        try:   
            Outcome_site = piont_take_cannula.work(path[:-4])      
        except :
            output_image,Outcome_site,image,img0=find_line_take_cannula_integration(path[:10] ,copy.deepcopy(image),copy.deepcopy(img0))
        

        dst6 = cv.imread('image/'+path[:-4]+'/'+'Ksvd.png',0)
        dst5 = cv.imread('image/'+path[:-4]+'/'+'Non-Local_Means.png',0)
        dst4 = cv.imread('image/'+path[:-4]+'/'+'Yaroslavsky.png',0)
        dst3 = cv.imread('image/'+path[:-4]+'/'+'Total_Variation.png',0)
        dst2 = cv.imread('image/'+path[:-4]+'/'+'Anisotropic.png',0)
        dst1 = cv.imread('image/'+path[:-4]+'/'+'Gaussian.png',0)
        dst0 = cv.imread('image/'+path[:-4]+'/'+'original.png',0)
        
        x_labels = ['Original', 'Gaussian', 'Anisotropic','Total_Variation','Yaroslavsky','Non_Local','Ksvd_method']
        
        Casing_profile= Convex_Hull(Outcome_site)                     # 提取凸包顶点
        dst_back_gray_all, _=Polygon_extraction(img0,Casing_profile)   # 提取红外套管区域图像，灰度图
        dst_back_rgb, _=Polygon_extraction(image,Casing_profile) # 提取红外套管区域图像，红外图
        Outcome_site,Casing_profile,dst_back_gray_all,dst_back_rgb,image = cut_image_conter(Outcome_site,Casing_profile,dst_back_gray_all,dst_back_rgb,image)
        dst7,_=two_to_one.anomalydetect(copy.deepcopy(dst_back_gray_all))
        temp_try_list["tr_pro_diff"][path[:-4]] = {}  # 初始化子字典
        temp_try_list["loca_small_mean_diff"][path[:-4]] = {}  # 初始化子字典
        temp_try_list["loca_small_max_diff"][path[:-4]] = {}  # 初始化子字典
        print(path)
        for cut_number in cut_list:
            x_number = cut_number[1]
            y_number = cut_number[0]
            loca_small_mean_diff =[]
            loca_small_max_diff = []
            tr_pro_diff = []
            piont_up_mid_down = Segmented_analysis(Outcome_site,x_number,y_number)
            
            # 原图
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst0),piont_up_mid_down,k,b,x_number,y_number)
            Actual_temperature = np.max(Small_area_max_temp)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
       
            # Gaussian Filtering
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst1),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            print('Gaussian',Small_area_temp)
            # Anisotropic Filtering
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst2),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            print('Anisotropic',Small_area_temp)
            # Total Variation Minimization
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst3),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            print('Total',Small_area_temp)
            
            # Yaroslavsky Filtering
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst4),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            print('Yaroslavsky',Small_area_temp)
            
            # NLMeans
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst5),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            print('NLMeans',Small_area_temp)
           
            # Ksvd method
            # some parameters
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst6),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            print('Ksvd',Small_area_temp)
            
            
            # 提取最大值
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst0),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp),3))
            else:
                loca_small_mean_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))),3))
             
            
            # 异常检测
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst7),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                tr_pro_diff.append(round(Actual_temperature-max(Small_area_max_temp), 3))
                        
            # temp_try_list["tr_pro_diff"][path[:-4]][str(y_number)+'*'+str(x_number)] = tr_pro_diff
            temp_try_list["loca_small_mean_diff"][path[:-4]][str(y_number)+'*'+str(x_number)] = loca_small_mean_diff
            # temp_try_list["loca_small_max_diff"][path[:-4]][str(y_number)+'*'+str(x_number)] = loca_small_max_diff

    # with open('Dir_mode_list_H.txt', 'w') as file:
        # for local, params in temp_try_list.items():
            # file.write(f"{local}: {params}\n")    
            
            
          
        
        
    
    
    
      
