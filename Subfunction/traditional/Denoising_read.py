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
from tqdm import tqdm 
from Subfunction.traditional.take_your_need import find_line_take_cannula ,piont_take_cannula 
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
           
    
    for itme in Small_area_temp_piont :          
        local= []   
        itme= Convex_Hull(itme)  
        dst_gray_small_area, _=Polygon_extraction(copy.deepcopy(image_gray),itme)
        nonzero_coords = np.transpose(np.nonzero(dst_gray_small_area))     

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
    model.load_state_dict(torch.load('Data/'+model_name))
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





def save_to_json(data_dict, file_name):
    with open(file_name, 'w') as json_file:
        json.dump(data_dict, json_file)
def load_from_json(file_name):
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)
    return data
     
def Denoising_read():
    piont_take_cannula.save_argument_to_txt()     
    find_line_take_cannula.save_argument_to_txt()               
    
    path_list = os.listdir('Data/Raw_data/')
    temp_try_list = { "loca_small_mean_diff": {}, "loca_small_max_diff": {}}
    temp_try_list["loca_small_mean_diff"] = {}
    temp_try_list["loca_small_max_diff"] = {}
        
        
    list_number=[[2,5],[2,10]]    
    
    Outcome_site_list = load_from_json('Data/Generated_data/traditional_denoising_write.json')
    for path in tqdm(path_list): 
        max_tmp , min_tmp = model_predict('Data/Raw_data/'+path)
        k , b =  get_regression(max_tmp,min_tmp)
        
        Outcome_site = Outcome_site_list[path[:-4]]
        
        temp_try_list["loca_small_mean_diff"][path[:-4]] = {}  # 初始化子字典
        temp_try_list["loca_small_max_diff"][path[:-4]] = {}  # 初始化子字典
        dst0 = cv.imread('Data/Generated_data/Traditional_denoising/'+path[:-4]+'/'+'original.png',0)
        dst1 = cv.imread('Data/Generated_data/Traditional_denoising/'+path[:-4]+'/'+'Gaussian.png',0)
        dst2 = cv.imread('Data/Generated_data/Traditional_denoising/'+path[:-4]+'/'+'Anisotropic.png',0)
        dst3 = cv.imread('Data/Generated_data/Traditional_denoising/'+path[:-4]+'/'+'Total_Variation.png',0)
        dst4 = cv.imread('Data/Generated_data/Traditional_denoising/'+path[:-4]+'/'+'Yaroslavsky.png',0)
        dst5 = cv.imread('Data/Generated_data/Traditional_denoising/'+path[:-4]+'/'+'Non-Local_Means.png',0)
        dst6 = cv.imread('Data/Generated_data/Traditional_denoising/'+path[:-4]+'/'+'Ksvd.png',0)
        filtered_image = cv.imread('Data/Generated_data/Traditional_denoising/'+path[:-4]+'/'+'filtered_image.png',0) 
        for numb in list_number:
            x_number = numb[1]
            y_number = numb[0]
            loca_small_mean_diff = []
            loca_small_max_diff = []
            piont_up_mid_down = Segmented_analysis(copy.deepcopy(Outcome_site),x_number,y_number)
        
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst0),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                 
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                 

            # Gaussian Filtering
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst1),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                 
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                 
            
            # Anisotropic Filtering
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst2),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                 
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                 
            
            # Total Variation Minimization
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst3),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                 
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                 
            
            # Yaroslavsky Filtering
            
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst4),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                 
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                 
            
            # NLMeans
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst5),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                 
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                 
            
            # Ksvd method
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst6),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                 
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                     
            # 提取最大值
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(dst0),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp),3))
            else:
                loca_small_mean_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))),3))
    
            # 异常检测
            Small_area_temp,Small_area_max_temp,Small_area_piont = small_area_divide(copy.deepcopy(filtered_image),piont_up_mid_down,k,b,x_number,y_number)
            if y_number == 1 :
                loca_small_mean_diff.append(round(max(Small_area_temp)-min(Small_area_temp), 3))
                loca_small_max_diff.append(round(max(Small_area_max_temp)-min(Small_area_max_temp), 3))
                 
            else:
                Small_area_temp_max0 = round(max(Small_area_temp[0::2]) - min(Small_area_temp[0::2]), 3)
                Small_area_temp_max1 = round(max(Small_area_temp[1::2]) - min(Small_area_temp[1::2]), 3)
                loca_small_mean_diff.append(max(Small_area_temp_max0,Small_area_temp_max1))
                loca_small_max_diff.append(round(max((max(Small_area_max_temp[0::2]) - min(Small_area_max_temp[0::2])),(max(Small_area_max_temp[1::2]) - min(Small_area_max_temp[1::2]))), 3))
                 
            temp_try_list["loca_small_mean_diff"][path[:-4]][str(y_number)+'*'+str(x_number)] = loca_small_mean_diff
            
    with open('Data/Generated_data/Traditional_denoising/traditional_denoising_temp.txt', 'w') as file:
        for local, params in temp_try_list.items():
            file.write(f"{local}: {params}\n")    
            
            
          
