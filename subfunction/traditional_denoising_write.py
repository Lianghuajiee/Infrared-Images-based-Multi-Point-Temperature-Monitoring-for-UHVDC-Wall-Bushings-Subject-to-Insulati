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
from subfunction.Traditional_denoising_model import NLM
from subfunction.Anomaly_detection import two_to_one
from subfunction.take_your_need import cut_image_conter1,find_line_take_cannula_integration ,piont_take_cannula
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
            colors=(0,255,0)    
            cv.line(image_gray, mid_area[0], mid_area[1], colors, 1, cv.LINE_AA)
            cv.line(image_gray, mid_area[2], mid_area[3], colors, 1, cv.LINE_AA)
            cv.line(image_gray, mid_area[0], mid_area[2], colors, 1, cv.LINE_AA)
            cv.line(image_gray, mid_area[1], mid_area[3], colors, 1, cv.LINE_AA)
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
     
     
def traditional_denoising_write():

    path_list = os.listdir('Data/Raw_data/')
    Outcome_site_list = {}
    if not os.path.exists('Data/Generated_data/Traditional_denoising'):
        os.makedirs('Data/Generated_data/Traditional_denoising')
        
    for path in tqdm(path_list):
   
        image = cv.imread('Data/Raw_data/'+path)                            # 图片输入
        img0 = cv.cvtColor(image,cv.COLOR_BGR2GRAY)         # 将彩色图片转换为灰度图片
        try:   
            Outcome_site = piont_take_cannula.work(path[:-4],1)      
        except :
            Outcome_site,smooth_size=find_line_take_cannula_integration(path[:10] ,copy.deepcopy(image),copy.deepcopy(img0),1)
            Outcome_site = Outcome_site + smooth_size
            
        Casing_profile= Convex_Hull(Outcome_site)                     # 提取凸包顶点
        dst_back_gray_all, _=Polygon_extraction(img0,Casing_profile)   # 提取红外套管区域图像，灰度图
        dst_back_rgb, _=Polygon_extraction(image,Casing_profile) # 提取红外套管区域图像，红外图
        Outcome_site1,Casing_profile1,dst_back_gray_all,dst_back_rgb,image = cut_image_conter1(copy.deepcopy(Outcome_site),copy.deepcopy(Casing_profile),dst_back_gray_all,dst_back_rgb,image)
        img0 = cv.cvtColor(image,cv.COLOR_BGR2GRAY)         # 将彩色图片转换为灰度图片
     
        Outcome_site_list[path[:-4]]= [item.tolist() for item in Outcome_site1]
        filtered_image,local2=two_to_one.anomalydetect(copy.deepcopy(dst_back_gray_all))
        cv.imwrite('Data/Generated_data/Traditional_denoising/'+path[:-4]+'.png', dst_back_gray_all)
        cv.imwrite('Data/Generated_data/Traditional_denoising/'+path[:-4]+'_image.png', image) 
        NLM_main = NLM(copy.deepcopy(img0))
        dst = []
        dst1,_ = NLM_main.Gaussian_Filtering(copy.deepcopy(NLM_main.image), dst, 7, 1.4)
        dst2,_ = NLM_main.Anisotropic_Filtering(copy.deepcopy(NLM_main.image), dst, 1, 5, 0.1)
        dst3,_ = NLM_main.Total_Variation_Minimization(copy.deepcopy(NLM_main.image),dst,1,0.01)
        dst4,_ = NLM_main.Yaroslavsky_Filtering(copy.deepcopy(NLM_main.image),dst,3,0.2)
        dst5,_ = NLM_main.NLMeans(copy.deepcopy(NLM_main.image),dst,10,3,1.5)
        dst6,_ = NLM_main.K_SVD_main(copy.deepcopy(NLM_main.image), dict_size=500, sparsity=2,window_stride = 5,ksvd_iter=1)
        dst6, _=Polygon_extraction(dst6,Casing_profile)
        dst5, _=Polygon_extraction(dst5,Casing_profile)
        dst4, _=Polygon_extraction(dst4,Casing_profile)
        dst3, _=Polygon_extraction(dst3,Casing_profile)
        dst2, _=Polygon_extraction(dst2,Casing_profile)
        dst1, _=Polygon_extraction(dst1,Casing_profile)
        filtered_image,local2=two_to_one.anomalydetect(copy.deepcopy(dst_back_gray_all))
        x_labels = ['original', 'Gaussian', 'Anisotropic','Total_Variation','Yaroslavsky','Non_Local','Ksvd_method', 'ours']
        cv.imwrite('Data/Generated_data/Traditional_denoising/'+path[:-4]+'_Gaussian.png', dst1)
        cv.imwrite('Data/Generated_data/Traditional_denoising/'+path[:-4]+'_Anisotropic.png', dst2)
        cv.imwrite('Data/Generated_data/Traditional_denoising/'+path[:-4]+'_Total_Variation.png', dst3)
        cv.imwrite('Data/Generated_data/Traditional_denoising/'+path[:-4]+'_Yaroslavsky.png', dst4)
        cv.imwrite('Data/Generated_data/Traditional_denoising/'+path[:-4]+'_Non-Local_Means.png', dst5)
        cv.imwrite('Data/Generated_data/Traditional_denoising/'+path[:-4]+'_Ksvd.png', dst6)
        cv.imwrite('Data/Generated_data/Traditional_denoising/'+path[:-4]+'_filtered_image.png', filtered_image) 
    save_to_json(Outcome_site_list, 'Data/Generated_data/enhance_Outcome_site_list.json')   
     