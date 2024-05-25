from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns

def Statistical_quantity_normel(loca_small_mean_diff,line):
    loca_small_mean_diff = np.array(loca_small_mean_diff)
    mask = loca_small_mean_diff <  line
    count_per_column = np.sum(mask, axis=0)   
    
    return count_per_column
  
def Statistical_quantity_Breakdown(data_Fault_image,line):
    data_Fault_image = np.array(data_Fault_image)
    mask = data_Fault_image >=  line
    count_per_column = np.sum(mask, axis=0)   
    return count_per_column        
    
def get_acc (name_txt):
    nan = 999
    with open(name_txt, 'r') as file:
        tr_pro_diff_normal = []
        tr_pro_diff_Breakdown = []
        loca_mean = []
        loca_max = []
        loca_small_mean_Breakdown = []
        loca_small_max_Breakdown = []
        for line in file:
            if  'loca_small_mean_diff' in line[:35] :
                line = line [len('loca_small_mean_diff')+2:]
                data = eval(line)
                data_2_5=[]
                data_2_10=[]
                loca_small_mean_Breakdowndata_2_5 = []
                loca_small_mean_Breakdowndata_2_10 = []
                for key, values in data.items():
                    if key != 'local_15_0 (9)' :
                        data_2_5.append(values['2*5'])
                        data_2_10.append(values['2*10'])
                    else :
                        loca_small_mean_Breakdowndata_2_5.append(values['2*5'])
                        loca_small_mean_Breakdowndata_2_10.append(values['2*10'])
                record_tm = 2.7
                loca_mean.append(np.array(Statistical_quantity_normel(data_2_5,record_tm)))
                loca_mean.append(np.array(Statistical_quantity_normel(data_2_10,record_tm)))
                loca_mean = np.array(loca_mean)
                loca_small_mean_Breakdown.append(np.array(Statistical_quantity_Breakdown(loca_small_mean_Breakdowndata_2_5,record_tm)))
                loca_small_mean_Breakdown.append(np.array(Statistical_quantity_Breakdown(loca_small_mean_Breakdowndata_2_10,record_tm)))
                loca_small_mean_Breakdown = np.array(loca_small_mean_Breakdown)
        if loca_small_mean_Breakdown[0] == 0 :
            return_nenber = 0
        else:
            return_nenber = np.round((loca_mean[0]+loca_small_mean_Breakdown[0])/161 ,3)
    return return_nenber





def matrix_two():
    model_list = ['One_svm', 'IsolationForest','EllipticEnvelope','DBSCAN', 'PCA','KMeans','HBOS','Z_Score','LocalOutlierFactor','Autoencoder']
    acc_list = {}
    number_list = []
    for list1 in model_list:
        acc_list[list1] = {}
    for list1 in model_list:
        for list2 in model_list:
            if list1 == list2 :
                acc_list[list1][list2] = get_acc('Data/Generated_data/one_acc/'+list1+'.txt')
            else:
                try:
                    acc_list[list1][list2] = get_acc('Data/Generated_data/two_acc//'+list1+'_'+list2+'.txt')
                    acc_list[list2][list1] = acc_list[list1][list2]
                except:
                    acc_list[list1][list2] = get_acc('Data/Generated_data/two_acc/'+list2+'_'+list1+'.txt')       
                    acc_list[list2][list1] = acc_list[list1][list2]
    acc_matrix = np.zeros((len(model_list), len(model_list)))
    for i, model1 in enumerate(model_list):
        for j, model2 in enumerate(model_list):
            if model1 in acc_list and model2 in acc_list[model1]:
                acc_matrix[i, j] = acc_list[model1][model2]
    plt.figure(figsize=(10, 8))
    plt.imshow(acc_matrix, interpolation='nearest', cmap='coolwarm')
    plt.colorbar()
    model_list = ['OneClassSVM', 'IsolationForest','EllipticEnvelope','DBSCAN', 'PCA','KMeans','HBOS','Z_Score','LocalOutlierFactor','Autoencoder']
    plt.xticks(np.arange(len(model_list)), model_list, rotation=90)
    plt.yticks(np.arange(len(model_list)), model_list)
    for i in range(len(model_list)):
        for j in range(len(model_list)):
            plt.text(j, i, f'{acc_matrix[i, j]:.2f}', ha='center', va='center', color='white')
    plt.title('Accuracy Matrix')
    plt.xlabel('Models')
    plt.ylabel('Models')
    plt.savefig('result/matrix.png') 
