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
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.image as mpimg
import time
import math
import json  
from PIL import Image as IMG

from Subfunction.Anomaly_detection import Model_selection
from Subfunction.take_your_need import find_line_take_cannula_integration ,piont_take_cannula ,cut_image_conter,find_line_take_cannula
import copy

np.set_printoptions(threshold=1000000)

def Convex_Hull (points):  #凸包顶点
    hull = ConvexHull(points)

    return points[hull.vertices]

    

def Polygon_extraction(image,Outcome_site):
    Outcome_site = np.array([Outcome_site])
    mask = np.zeros(image.shape[:2], np.uint8)
    cv.polylines(mask, Outcome_site, 1, 255,thickness=1)    # 描绘边缘
    cv.fillPoly(mask, Outcome_site, 255)    # 填充
    dst_back = cv.bitwise_and(image, image, mask=mask)
    bg = np.ones_like(image, np.uint8) * 255
    cv.bitwise_not(bg, bg, mask=mask)  # bg的多边形区域为0，背景区域为255
    dst_white = bg + dst_back
    # cv.imshow("dst_white.jpg", dst_white)
    return dst_back, dst_white
 


    
def get_tmp (k , b , pixel):   #获取温度
    return round(pixel * k + b, 2)
    


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

                      
def get_modal_data():
    piont_take_cannula.save_argument_to_txt()     
    find_line_take_cannula.save_argument_to_txt()               
    path_list = os.listdir('Data/Raw_data/')
    
    if not os.path.exists('Data/Generated_data/Preconditioning_data/'): 
        os.mkdir('Data/Generated_data/Preconditioning_data/')    
    if not os.path.exists('Data/Generated_data/Preconditioning_data_image/'): 
        os.mkdir('Data/Generated_data/Preconditioning_data_image/')    
    model_list = ['One_svm', 'IsolationForest','EllipticEnvelope','DBSCAN', 'PCA','KMeans','HBOS','Z_Score','LocalOutlierFactor','Autoencoder']
    for num,path in enumerate(path_list): 
        image = cv.imread('Data/Raw_data/'+path)                     # 图片输入
        img0 = cv.cvtColor(image,cv.COLOR_BGR2GRAY)         # 将彩色图片转换为灰度图片
        try:   
            Outcome_site = piont_take_cannula.work(path[:-4],1)      
        except :
            Outcome_site,si=find_line_take_cannula_integration(path[:10] ,copy.deepcopy(image),copy.deepcopy(img0),1)
            Outcome_site = Outcome_site+ si
        Casing_profile= Convex_Hull(Outcome_site)                     # 提取凸包顶点
        dst_back_gray_all, _=Polygon_extraction(img0,Casing_profile)   # 提取红外套管区域图像，灰度图
        dst_back_rgb, _=Polygon_extraction(image,Casing_profile) # 提取红外套管区域图像，红外图
        Outcome_site,Casing_profile,dst_back_gray_all,dst_back_rgb,image = cut_image_conter(Outcome_site,Casing_profile,dst_back_gray_all,dst_back_rgb,image)
        cv.imwrite('Data/Generated_data/Preconditioning_data_image/'+path[:-4]+'_dst_back_rgb.png',copy.deepcopy(dst_back_rgb))
        cv.imwrite('Data/Generated_data/Preconditioning_data_image/'+path[:-4]+'_dst_back_gray_all.png',copy.deepcopy(dst_back_gray_all))
        cv.imwrite('Data/Generated_data/Preconditioning_data_image/'+path[:-4]+'_image.png',copy.deepcopy(image)) 
        for model in model_list:    
            print(num,path,model)
            Preconditioning_list = {"data_list": [], "file_area_list": [], "tsne_list": [], "filtered_list": []}
            Preconditioning_list['data_list'] = []
            Preconditioning_list['file_area_list'] =[]
            Preconditioning_list['tsne_list'] = []
            Preconditioning_list['filtered_list'] = []
            Preconditioning_list['Outcome_site'] = []
            data = []
            file_area = []
            image_tsne = []
            filtered_e = []
            data , file_area , tsne_data , filtered_e=Model_selection.model_every_one(copy.deepcopy(dst_back_gray_all) , model)
            tsne = TSNE(n_components=2, random_state=0)
            image_tsne = tsne.fit_transform(tsne_data)
            Preconditioning_list['data_list'] = np.array(data).tolist()
            Preconditioning_list['file_area_list'] = np.array(file_area).tolist()
            Preconditioning_list['tsne_list'] = np.array(image_tsne).tolist()
            Preconditioning_list['filtered_list']= np.array(filtered_e).tolist()
            Preconditioning_list['Outcome_site']= np.array(Outcome_site).tolist()
            
            with open('Data/Generated_data/Preconditioning_data/'+model+'_'+path[:-4]+'.json', 'w') as json_file:
                json.dump(Preconditioning_list, json_file)    
            
         
            
          
            
        
   

                
                
            
            
            
            
            
            
            
            
            
            
            
        
        
        
          
                            