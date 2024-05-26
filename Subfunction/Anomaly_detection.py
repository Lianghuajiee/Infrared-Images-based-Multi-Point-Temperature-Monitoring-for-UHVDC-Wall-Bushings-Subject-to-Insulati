from sklearn.ensemble import IsolationForest
from sklearn import svm
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from pyod.models.hbos import HBOS
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
import copy
import random
from sklearn.utils import check_random_state
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
rng = check_random_state(42)
np.random.seed(42)
random.seed(42)

class Various_parameters: 
    def build(model_type,length):
        if model_type=="IsolationForest":    #reshape
            rng = np.random.RandomState(50000)
            m=IsolationForest(max_samples=length, random_state=42,contamination='auto', max_features=1,bootstrap=True,n_estimators=32)
        elif model_type=="One_svm":
            m=svm.OneClassSVM(nu=0.2, kernel="poly",gamma='scale')#nu取0-0.5, 核函数kernel可选择'linear''poly''rbf'， gamma可选择'scale''auto'
        elif model_type=="EllipticEnvelope":    
            m=EllipticEnvelope(contamination=0.06)
        elif model_type=="DBSCAN":    
            m=DBSCAN(eps=1, min_samples=500)
        elif model_type=="PCA":    
            m=PCA()
        elif model_type=="KMeans":    
            m=KMeans(n_clusters=2, n_init=10)
        elif model_type=="HBOS":    
            m=HBOS(contamination=0.2)
        elif model_type=="Z_Score":    
            m=stats
        elif model_type=="LocalOutlierFactor":
            m=LocalOutlierFactor(n_neighbors=4,metric= 'cosine', contamination=0.06,algorithm='brute')
        else:
            print("model_type error")
            exit()
        return m
    def count_number(prd,model,shape1_number):
        count_0 = 0
        count_1 = 0
        if model == "Z_Score" or model == "PCA" :
        
            record_list =np.array(prd)
            record_list = record_list.reshape(1, -1)[0]
            record_list = list(record_list)
            record_list = sorted(record_list)
         
            number = record_list[int(len(record_list)*0.02)]  
            record_list =np.array(prd)
            record_list = record_list.reshape(1, -1)[0]
            record_list = list(record_list)
            
            for i in range(len(record_list)):
                if  record_list[i] < number :
                    record_list[i] = 1
                else:
                    record_list[i] = 0
            record_list =np.array(record_list)        
            prd = record_list.reshape(-1, shape1_number)
        
        prd = np.array(prd)

        count_0 = np.sum(prd == 0)
        count_1 = np.sum(prd == 1)
        count_11 = np.sum(prd == -1)
        if count_0 == 0:
            if count_11 > count_1:       #   0异常       1正常
                prd[prd == 1] = 0
                prd[prd == -1] = 1
               
            else:
                prd[prd == -1] = 0
        elif count_1 == 0:    
            if count_0 > count_11:       #   0异常       1正常
                prd[prd == 0] = 1
                prd[prd == -1] = 0
               
            else:
                prd[prd == -1] = 1
        else:    
            if count_0 > count_1:       #   0异常       1正常
                prd[prd == 0] = 2
                prd[prd == 1] = 0
                prd[prd == 2] = 1
        return prd

    def anomalydetect(image_gray,mode):
        file_number=[]
        file_area=[]
        filtered_image = np.zeros_like(image_gray, dtype=np.float32)   
        if mode == 'one_to_one':
            shape1_number=1
            for i in range(0, image_gray.shape[0]):
                for j in range(0, image_gray.shape[1]):
                    if image_gray[i, j] != 0:
                        filtered_image[i, j] = image_gray[i, j]
                        file_number.append([image_gray[i, j]])
                        file_area.append([i,j])
        elif mode == 'two_to_two' or mode ==  'two_to_one':  
            shape1_number=4
            for i in range(0, image_gray.shape[0]-1):             #2*2 kel
                for j in range(0, image_gray.shape[1]-1):
                    if image_gray[i, j] != 0:
                        neighbors = image_gray[i:i+2, j:j+2]
                        neighbors = neighbors[neighbors != 0]
                        if  len(neighbors) == 4 :
                            filtered_image[i, j] = image_gray[i, j]
                            file_number.append(neighbors)
                            file_area.append([i,j])
        elif mode == 'three_to_three' or mode ==  'three_to_one':  
            shape1_number=9
            for i in range(1, image_gray.shape[0]-1):             #3*3 kel
                for j in range(1, image_gray.shape[1]-1):
                    if image_gray[i, j] != 0:
                        neighbors = image_gray[i-1:i+2, j-1:j+2]
                        neighbors = neighbors[neighbors != 0]
                        if  len(neighbors) == 9 :
                            filtered_image[i, j] = image_gray[i, j]
                            file_number.append(neighbors)
                            file_area.append([i,j])
        elif mode == 'four_to_one' :  
            shape1_number=4**2
            for i in range(0, image_gray.shape[0]-3):             #3*3 kel
                for j in range(0, image_gray.shape[1]-3):
                    if image_gray[i, j] != 0:
                        neighbors = image_gray[i:i+4, j:j+4]
                        neighbors = neighbors[neighbors != 0]
                        if  len(neighbors) == shape1_number :
                            filtered_image[i, j] = image_gray[i, j]
                            file_number.append(neighbors)
                            file_area.append([i,j])
        elif mode == 'five_to_one':  
            shape1_number=5**2
            for i in range(2, image_gray.shape[0]-2):             #3*3 kel
                for j in range(2, image_gray.shape[1]-2):
                    if image_gray[i, j] != 0:
                        neighbors = image_gray[i-2:i+3, j-2:j+3]
                        neighbors = neighbors[neighbors != 0]
                        if  len(neighbors) == shape1_number :
                            filtered_image[i, j] = image_gray[i, j]
                            file_number.append(neighbors)
                            file_area.append([i,j])
        file_number=np.array(file_number)   
        file_area=np.array(file_area) 
        filtered_image = np.uint8(filtered_image) 
        data_lst=np.array(file_number)
        model_type_lst=["One_svm","Z_Score"] 
        problem_point = []
        problem_point_is_ont = []
        pred1=[]
        pred2=[]
        pred3=[]
        
        for model in model_type_lst:
            if mode == 'one_to_one':
                m=Various_parameters.build(model,data_lst.shape[0]) #搭建模型     shape[1]
            else:    
                m=Various_parameters.build(model,data_lst.shape[1]) #搭建模型     shape[1]
            
            if model == "Z_Score":
                pred2 = stats.zscore(data_lst)
                pred2 = Various_parameters.count_number(pred2,model,shape1_number)
        
            else:
                m=Various_parameters.build(model,data_lst.shape[1]) #搭建模型
                m.fit(data_lst) #训练模型     #Z-Score不需要这个
                for data in data_lst:
                    try:
                        predict=m.predict(data.reshape(-1, 1))
                      
                    except:
                        predict=m.fit_predict(data.reshape(-1, 1))
                    if model=="One_svm":
                        pred1.append(predict)
                pred1 = Various_parameters.count_number(pred1,model,shape1_number)
            
            
        for i in range(len(pred1)):
            if not sum(pred1[i])==sum(pred2[i])==  shape1_number :     #sum(pred3[i]) == 
                i1=file_area[i,0]
                j1=file_area[i,1]
                if mode == 'one_to_one':
                    problem_point.append(file_area[i])
                elif mode == 'two_to_two' :
                    for nember in range(2): 
                        problem_point.append([i1,j1+nember])
                        problem_point.append([i1+1,j1+nember])
                elif mode == 'two_to_one':     
                    for pp in range(4): 
                        if not pred1[i][pp] == pred2[i][pp] == 1:
                      
                            if pp <= 1 : 
                                problem_point.append([i1,j1+pp])
                            else:
                                u = pp-2 
                                problem_point.append([i1+1,j1+u])
                elif mode == 'three_to_three' :
                    j1 -= 1 
                    for nember in range(3): 
                        problem_point.append([i1-1,j1+nember])
                        problem_point.append([i1,j1+nember])
                        problem_point.append([i1+1,j1+nember])
                elif mode == 'three_to_one':   
                    j1 -= 1 
                    for pp in range(9): 
                        if not pred1[i][pp] == pred2[i][pp] == 1:
                            if pp <= 2 : 
                                problem_point.append([i1+1,j1+pp])
                            elif pp <= 2+3 : 
                                u = pp-2 
                                problem_point.append([i1,j1+u])
                            elif pp <= 2+3+3 :     
                                u = pp-2-3 
                                problem_point.append([i1+1,j1+u])
                elif mode == 'four_to_one' :  
                    for pp in range(4**2): 
                        if not pred1[i][pp] == pred2[i][pp] ==1:
                            if pp <= 3 : 
                                problem_point.append([i1,j1+pp])
                            elif pp <= 3+4 : 
                                u = pp-3 
                                problem_point.append([i1+1,j1+u])
                            elif pp <= 3+8 :     
                                u = pp-3-4
                                problem_point.append([i1+2,j1+u])
                            elif pp <= 3+8+4 :     
                                u = pp-3-4-4
                                problem_point.append([i1+3,j1+u])    
                elif mode == 'five_to_one':                  
                    for pp in range(5**2): 
                        if not pred1[i][pp] == pred2[i][pp] == 1:
                            if pp <= 4 : 
                                problem_point.append([i1-2,j1+pp])
                            elif pp <= 4+5 : 
                                u = pp-4 
                                problem_point.append([i1-1,j1+u])
                            elif pp <= 4+5+5 :     
                                u = pp-4-5
                                problem_point.append([i1,j1+u])
                            elif pp <= 4+5+5+5 :     
                                u = pp-4-5-5
                                problem_point.append([i1+1,j1+u])                
                            elif pp <= 4+5+5+5+5 :     
                                u = pp-4-5-5-5
                                problem_point.append([i1+2,j1+u])                
                                
        problem_point = np.array(problem_point)    
      
                    
        for point in problem_point:
            filtered_image[point[0], point[1]] = 0
        local2= []   
        nonzero_coords = np.transpose(np.nonzero(filtered_image))     # 获取非零像素的坐标
        for coord in nonzero_coords:
            local2.append(filtered_image[coord[0], coord[1]])
        local2=sorted(local2)
        
        return filtered_image , local2
        
        


        


class two_to_one:
    def build(model_type,length):
        if model_type=="IsolationForest":    #reshape
            rng = np.random.RandomState(50000)
            m=IsolationForest(max_samples=length, random_state=42,contamination='auto', max_features=1,bootstrap=True,n_estimators=32)
        elif model_type=="One_svm":
            m=svm.OneClassSVM(nu=0.2, kernel="poly",gamma='scale')#nu取0-0.5, 核函数kernel可选择'linear''poly''rbf'， gamma可选择'scale''auto'
        elif model_type=="EllipticEnvelope":    
            m=EllipticEnvelope(contamination=0.06)
        elif model_type=="DBSCAN":    
            m=DBSCAN(eps=1, min_samples=500)
        elif model_type=="PCA":    
            m=PCA()
        elif model_type=="KMeans":    
            m=KMeans(n_clusters=2, n_init=10)
        elif model_type=="HBOS":    
            m=HBOS(contamination=0.2)
        elif model_type=="Z_Score":    
            m=stats
        elif model_type=="LocalOutlierFactor":
            m=LocalOutlierFactor(n_neighbors=4,metric= 'cosine', contamination=0.06,algorithm='brute')
        else:
            print("model_type error")
            exit()
        return m

    def count_number(prd,model):
        count_0 = 0
        count_1 = 0
        if model == "Z_Score" or model == "PCA" :
        
            record_list =np.array(prd)
            record_list = record_list.reshape(1, -1)[0]
            record_list = list(record_list)
            record_list = sorted(record_list)
         
            number = record_list[int(len(record_list)*0.02)]  
            record_list =np.array(prd)
            record_list = record_list.reshape(1, -1)[0]
            record_list = list(record_list)
            
            for i in range(len(record_list)):
                if  record_list[i] < number :
                    record_list[i] = 1
                else:
                    record_list[i] = 0
            record_list =np.array(record_list)        
            prd = record_list.reshape(-1, 4)
        
        prd = np.array(prd)

        count_0 = np.sum(prd == 0)
        count_1 = np.sum(prd == 1)
        count_11 = np.sum(prd == -1)
        if count_0 == 0:
            if count_11 > count_1:       #   0异常       1正常
                prd[prd == 1] = 0
                prd[prd == -1] = 1
            else:
                prd[prd == -1] = 0
        elif count_1 == 0:    
            if count_0 > count_11:       #   0异常       1正常
                prd[prd == 0] = 1
                prd[prd == -1] = 0
            else:
                prd[prd == -1] = 1
        else:    
            if count_0 > count_1:       #   0异常       1正常
                prd[prd == 0] = 2
                prd[prd == 1] = 0
                prd[prd == 2] = 1
            
          
        return prd
        
    def anomalydetect(image_gray):            # two_to_one  
        file_number=[]
        file_area=[]
        filtered_image = np.zeros_like(image_gray, dtype=np.float32)   
        # 对图像中的非零像素进行均值滤波
        for i in range(1, image_gray.shape[0]-1):
            for j in range(1, image_gray.shape[1]-1):
                if image_gray[i, j] != 0:
                    neighbors = image_gray[i:i+2, j:j+2]
                    neighbors = neighbors[neighbors != 0]
                    if  len(neighbors) == 4 :
                        filtered_image[i, j] = image_gray[i, j]
                        file_number.append(neighbors)
                        file_area.append([i,j])
        file_number=np.array(file_number)   
        file_area=np.array(file_area) 
        filtered_image = np.uint8(filtered_image) 
        
        data_lst=np.array(file_number)
        number_list=[]
        output_pred = []

        model_type_lst=["One_svm","Z_Score"]
        problem_point=[]
        pred1=[]
        pred2=[]
        pred3=[]
        
        for model in model_type_lst:
            if model == "Z_Score":
                pred1 = stats.zscore(data_lst)
                output_pred_1 = two_to_one.count_number(pred1,model)
            else:
                m=two_to_one.build(model,data_lst.shape[1]) 
                m.fit(data_lst) 
                for data in data_lst:
                    try:
                        try:
                            predict=m.predict(data.reshape(-1, 1))
                        except:
                            try:
                                predict=m.fit_predict(data.reshape(-1, 1))
                            except:    
                                predict=m.fit_transform(data.reshape(-1, 1))
                        if model=="One_svm":
                            pred2.append(predict)
                    except:
                        pred2.append([1,1,1,1])          
                output_pred_2 = two_to_one.count_number(pred2,model)  
        
        
        output_pred_2 = np.array(output_pred_2)
        output_pred_1 = np.array(output_pred_1)
        output_pred_2 = output_pred_2.reshape(-1,4)
        output_pred_1 = output_pred_1.reshape(-1,4)    
        for i in range(output_pred_2.shape[0]):
            if not sum(output_pred_2[i])==   sum(output_pred_1[i])==  4 :   #   0异常       1正常    sum(output_pred_1[i])==    
                i1=file_area[i,0]
                j1=file_area[i,1]
                for pp in range(4): 
                    if not output_pred_2[i,pp] == output_pred_1[i,pp] ==  1:     #均正常才保留        == output_pred_1[i,pp]      
                        if pp <= 1 : 
                            problem_point.append([i1,j1+pp])
                        else:
                            u = pp-2 
                            problem_point.append([i1+1,j1+u])
        problem_point = np.array(problem_point)    
      
        for point in problem_point:
            filtered_image[point[0], point[1]] = 0
        local2= []   
    
        return filtered_image , local2
  
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
        
class Autoencoder_m(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder_m, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded        
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x       

        
class Model_selection:
    
    def build(model_type,length):
        if model_type=="IsolationForest":  
            rng = np.random.RandomState(50000)
            m=IsolationForest(max_samples=length, random_state=42,contamination='auto', max_features=1,bootstrap=True,n_estimators=32)
        elif model_type=="One_svm":
            m=svm.OneClassSVM(nu=0.2, kernel="poly",gamma='scale')
        elif model_type=="EllipticEnvelope":    
            m=EllipticEnvelope(contamination=0.06)
        elif model_type=="DBSCAN":    
            m=DBSCAN(eps=1, min_samples=500)
        elif model_type=="PCA":    
            m=PCA()
        elif model_type=="KMeans":    
            m=KMeans(n_clusters=2, n_init=10)
        elif model_type=="HBOS":    
            m=HBOS(contamination=0.2)
        elif model_type=="Z_Score":    
            m=stats
        elif model_type=="LocalOutlierFactor":
            m=LocalOutlierFactor(n_neighbors=4,metric= 'cosine', contamination=0.06,algorithm='brute')
        else:
            print("model_type error")
            exit()
        return m
    def count_number(prd,model):
        count_0 = 0
        count_1 = 0
        if model == "Z_Score" or model == "PCA" :
        
            record_list =np.array(prd)
            record_list = record_list.reshape(1, -1)[0]
            record_list = list(record_list)
            record_list = sorted(record_list)
         
            number = record_list[int(len(record_list)*0.02)]  
            record_list =np.array(prd)
            record_list = record_list.reshape(1, -1)[0]
            record_list = list(record_list)
            
            for i in range(len(record_list)):
                if  record_list[i] < number :
                    record_list[i] = 1
                else:
                    record_list[i] = 0
            record_list =np.array(record_list)        
            prd = record_list.reshape(-1, 4)
        
        prd = np.array(prd)
        count_0 = np.sum(prd == 0)
        count_1 = np.sum(prd == 1)
        count_11 = np.sum(prd == -1)
        if count_0 == 0:
            if count_11 > count_1:       #   0异常       1正常
                prd[prd == 1] = 0
                prd[prd == -1] = 1
            else:
                prd[prd == -1] = 0
        elif count_1 == 0:    
            if count_0 > count_11:       #   0异常       1正常
                prd[prd == 0] = 1
                prd[prd == -1] = 0
            else:
                prd[prd == -1] = 1
        else:    
            if count_0 > count_1:       #   0异常       1正常
                prd[prd == 0] = 2
                prd[prd == 1] = 0
                prd[prd == 2] = 1
        return prd
    def model_every_one(image_gray,model):
        file_number=[]
        file_area=[]
        filtered_image = np.zeros_like(image_gray, dtype=np.float32)   
        for i in range(1, image_gray.shape[0]-1):
            for j in range(1, image_gray.shape[1]-1):
                if image_gray[i, j] != 0:
                    neighbors = image_gray[i:i+2, j:j+2]
                    neighbors = neighbors[neighbors != 0]
                    if  len(neighbors) == 4 :
                        filtered_image[i, j] = image_gray[i, j]
                        file_number.append(neighbors)
                        file_area.append([i,j])
        file_number=np.array(file_number)   
        file_area=np.array(file_area) 
        filtered_image = np.uint8(filtered_image) 
        
        data_lst=np.array(file_number)
        number_list=[]
        output_pred = []
        problem_point = []
        pred = []
        if model == "Z_Score":
            pred = stats.zscore(data_lst)
      
        elif model == "Autoencoder":
            all_data_tensor = torch.tensor(data_lst, dtype=torch.float32)
            use_model = Autoencoder_m(4, 2)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(use_model.parameters(), lr=0.001)

            num_epochs = 100
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = use_model(all_data_tensor)
                loss = criterion(outputs, all_data_tensor)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                reconstructed_data = use_model(all_data_tensor)
                mse = torch.mean((all_data_tensor - reconstructed_data)**2, dim=1).numpy()
            threshold = np.mean(mse) + 2*np.std(mse)
            pred111 = (mse < threshold).astype(int)
            
            for oo in pred111:
                if oo == 1 :
                    pred.append([1,1,1,1])
                elif oo == 0 :
                    pred.append([0,0,0,0])
                elif oo == -1 :
                    pred.append([-1,-1,-1,-1])
                else:
                    print("wrong")
                    exit()       
        else:
            m=Model_selection.build(model,data_lst.shape[1]) #搭建模型
            m.fit(data_lst) #训练模型     #Z-Score不需要这个

            for data in data_lst:
                try:
                    try:
                        predict=m.predict(data.reshape(-1, 1))

                    except:
                        try:
                            predict=m.fit_predict(data.reshape(-1, 1))
                        except:    
                            predict=m.fit_transform(data.reshape(-1, 1))
                    if model=="One_svm":
                        pred.append(predict)
                    elif model=="DBSCAN":
                        pred.append(predict)
                    elif model=="LocalOutlierFactor":
                        pred.append(predict)
                    elif model=="EllipticEnvelope":
                        pred.append(predict)
                    elif model=="PCA":
                        pred.append(predict.ravel())
                    elif model=="KMeans":
                        pred.append(predict)
                    elif model=="HBOS":
                        pred.append(predict)

                    elif model=="IsolationForest":
                        pred.append(predict)  
                except:
                    pred.append([1,1,1,1])  
        output_pred = Model_selection.count_number(pred,model)
        return output_pred , file_area  , data_lst ,filtered_image 
            
    def anomalydetect_one(pred1,file_area,filtered_image):
        number_list = []
        problem_point = []
        pred1 = np.array(pred1)
        pred1 = pred1.reshape(-1,4)
        for i in range(pred1.shape[0]):
            if not sum(pred1[i])== 4 :     # 
                i1=file_area[i,0]
                j1=file_area[i,1]
                number_list.append(0)
                for pp in range(4): 
                    if  pred1[i,pp] == 0 :  # 
                        if pp <= 1 : 
                            problem_point.append([i1,j1+pp])
                        else:
                            u = pp-2 
                            problem_point.append([i1+1,j1+u])
            else:
                number_list.append(1)                
        problem_point = np.array(problem_point)    
        for point in problem_point:
            filtered_image[point[0], point[1]] = 0
        return filtered_image , number_list
        
    def anomalydetect_two_abnormel(pred1,pred2,file_area,filtered_image):
        number_list = []
        problem_point = []
        pred1 = np.array(pred1)
        pred2 = np.array(pred2)
        pred1 = pred1.reshape(-1,4)
        pred2 = pred2.reshape(-1,4)
        
        for i in range(pred2.shape[0]):
            if not sum(pred1[i])== sum(pred2[i])==  4 :  
                i1=file_area[i,0]
                j1=file_area[i,1]
                number_list.append(0)
                for pp in range(4): 
                    if  pred1[i,pp] == pred2[i,pp] == 0 : 
                        if pp <= 1 : 
                            problem_point.append([i1,j1+pp])
                        else:
                            u = pp-2 
                            problem_point.append([i1+1,j1+u])
            else:
                number_list.append(1)                
        
            
        problem_point = np.array(problem_point)    
      
        for point in problem_point:
            filtered_image[point[0], point[1]] = 0
        return filtered_image , number_list    
   
    def anomalydetect_two_normel(pred1,pred2,file_area,filtered_image):
        number_list = []
        problem_point = []
        pred1 = np.array(pred1)
        pred2 = np.array(pred2)
        pred1 = pred1.reshape(-1,4)
        pred2 = pred2.reshape(-1,4)
        
        for i in range(pred2.shape[0]):
            if not sum(pred1[i])== sum(pred2[i])==  4 :    
                i1=file_area[i,0]
                j1=file_area[i,1]
                number_list.append(0)
                for pp in range(4): 
                    if  not pred1[i,pp] == pred2[i,pp] == 1 :
                        if pp <= 1 : 
                            problem_point.append([i1,j1+pp])
                        else:
                            u = pp-2 
                            problem_point.append([i1+1,j1+u])
            else:
                number_list.append(1)                
        problem_point = np.array(problem_point)    
        for point in problem_point:
            filtered_image[point[0], point[1]] = 0
        return filtered_image , number_list