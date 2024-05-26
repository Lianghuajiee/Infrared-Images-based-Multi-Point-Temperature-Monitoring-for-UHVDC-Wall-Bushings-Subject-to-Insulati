import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import json
def read_from_txt(name):
    param_dict = {}
    with open(name, 'r') as file:
        for line in file:
            split_index = line.find(':')  # Find the first colon
            local = line[:split_index].strip()
            params = line[split_index + 1:].strip()
            if 'nan' in params:
                params = params.replace('nan', '5')
            param_dict[local] = eval(params)  
            param_dict[local] = [ round(i,3 ) for i in param_dict[local]]
               
    return param_dict
    


def Statistical_quantity_normel(loca_small_mean_diff,line):
    
    loca_small_mean_diff = np.array(loca_small_mean_diff)
    mask = loca_small_mean_diff <  line
    count_per_column = np.sum(mask, axis=0)

    
    return count_per_column
def Statistical_quantity_Breakdown(data_Fault_image,line):
    data_Fault_image = np.array(data_Fault_image)
    mask = data_Fault_image >  line
    count_per_column = np.sum(mask, axis=0)
    
    
    return count_per_column  
nan = 999    
def Result_analysis_kernel():
    name_txt_list = os.listdir('Data/Generated_data/cut_list_acc/')
    temp_try_list = {}
    for name_txt in name_txt_list:
        with open( 'Data/Generated_data/cut_list_acc/'+name_txt, 'r') as file:
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
            temp_try_list[name_txt[:-4]] = {'Accuracy':np.round((loca_mean[0]+loca_small_mean_Breakdown[0])/161 ,3),'True Negative Rate':loca_small_mean_Breakdown[0],'True Positive  Rate':np.round((loca_mean[0])/160 ,3)}
            print(name_txt[:-4]+':',np.round((loca_mean[0]+loca_small_mean_Breakdown[0])/161 ,3))
    with open('Result/Impact_of_Kernel_Size_on_Accuracy.txt', 'w') as file:
        for local, params in temp_try_list.items():
            file.write(f"{local}: {params}\n")
            
            
            
def calculate_f1(loca_mean, loca_small_mean_Breakdown,q, i):
    TP = loca_mean[q, i]
    FN = 160 - loca_mean[q, i]
    FP = 1 - loca_small_mean_Breakdown[q, i]
    TN = loca_small_mean_Breakdown[q, i]
    
    if TP + FP == 0 or TP + FN == 0:
        return 0  # To handle the case of division by zero

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    if precision + recall == 0:
        return 0  # To handle the case of division by zero

    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1
            
def Result_analysis_4():
    with open( 'Data/Generated_data/Traditional_denoising/traditional_denoising_temp.txt', 'r') as file:
        temp_try_list={}
        loca_mean = []
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
        
        name_list=['Original','Gaussian','Anisotropic','Total_Variation','Yaroslavsky','Non-Local_Means','Ksvd','Take maximum','Our']
        for i , name in enumerate(name_list):
            temp_try_list[name] = {'Accuracy':np.round((loca_mean[0,i]+loca_small_mean_Breakdown[0,i])/161 ,3),'True Positive':loca_mean[0,i],'False Negative':160-loca_mean[0,i],'False Positive':1-loca_small_mean_Breakdown[0,i],'True Negative':loca_small_mean_Breakdown[0,i]}
            print(name+'_Accuracy:',temp_try_list[name])
        
    
    with open('Result/FIGURE_4_Accuracy.txt', 'w') as file:
        for local, params in temp_try_list.items():
            file.write(f"{local}: {params}\n")
            
def Result_analysis_6():
    with open( 'Data/Generated_data/Traditional_denoising/traditional_denoising_temp.txt', 'r') as file:
        temp_try_list={}
        loca_mean = []
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
        
        
        temp_try_list['Original'] ={'10_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,0,0),'20_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,1,0)}
        temp_try_list['Gaussian'] = {'10_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,0,1),'20_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,1,1)}
        temp_try_list['Anisotropic'] ={'10_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,0,2),'20_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,1,2)}
        temp_try_list['Total_Variation'] = {'10_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,0,3),'20_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,1,3)}
        temp_try_list['Yaroslavsky'] = {'10_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,0,4),'20_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,1,4)}
        temp_try_list['Non-Local_Means'] = {'10_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,0,5),'20_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,1,5)}
        temp_try_list['Ksvd'] = {'10_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,0,6),'20_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,1,6)}
        temp_try_list['Take maximum'] ={'10_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,0,7),'20_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,1,7)}

        temp_try_list['Our'] ={'10_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,0,8),'20_F1':calculate_f1(loca_mean,loca_small_mean_Breakdown,1,8)}
        
        
        print('Original_Accuracy:',temp_try_list['Original'])
        print('Gaussian_Accuracy:',temp_try_list['Gaussian'])
        print('Anisotropic_Accuracy:',temp_try_list['Total_Variation'])
        print('Total_Variation_Accuracy:',temp_try_list['Yaroslavsky'])
        print('Non-Local_Means_Accuracy:',temp_try_list['Non-Local_Means'])
        print('Ksvd_Accuracy:',temp_try_list['Ksvd'])
        print('Take maximum_Accuracy:',temp_try_list['Take maximum'])
        print('Our_Accuracy:',temp_try_list['Our'])
        
    with open('Result/FIGURE_6_Accuracy.txt', 'w') as file:
        for local, params in temp_try_list.items():
            file.write(f"{local}: {params}\n")
            
            