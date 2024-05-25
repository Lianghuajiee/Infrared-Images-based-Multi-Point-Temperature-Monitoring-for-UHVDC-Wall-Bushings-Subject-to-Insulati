from subfunction.two_model_normal import get_two_model_acc 
from subfunction.one_model_normal import get_one_model_acc 
from subfunction.get_modal_data import get_modal_data 
from subfunction.Anomaly_detection import Model_selection
from subfunction.matrix_two import matrix_two
from subfunction.main_kernel import main_kernel

from subfunction.traditional.Denoising_read import Denoising_read
from subfunction.traditional.Denoising_main import Denoising_main
import sys
import os
from subfunction.take_your_need import find_line_take_cannula ,piont_take_cannula 
from subfunction.Result_analysis import Result_analysis_kernel , Result_analysis_4 ,Result_analysis_6



if __name__ == "__main__":
    piont_take_cannula.save_argument_to_txt()     
    find_line_take_cannula.save_argument_to_txt()
    
    if not os.path.exists('Data/Generated_data'): 
        os.mkdir('Data/Generated_data')    
 
    if sys.argv[1]=="FIGURE_4":
        try:
            # Denoising_read()
            Result_analysis_4()
            print('===============================================')
            print('======FIGURE 4 matrix generation complete======')
            print('===============================================')
        except:
            Denoising_main()
            Denoising_read()
            Result_analysis_4()
            print('===============================================')
            print('======FIGURE 4 matrix generation complete======')
            print('===============================================')
    elif sys.argv[1]=="FIGURE_6":
        try:
            Result_analysis_6()
            print('===============================================')
            print('======FIGURE 6 matrix generation complete======')
            print('===============================================')
        except:
            Denoising_main()
            Denoising_read()
            Result_analysis_6()
            print('===============================================')
            print('======FIGURE 6 matrix generation complete======')
            print('===============================================')
            
            
    elif sys.argv[1]=="FIGURE_12":
        try:
            get_one_model_acc()
            get_two_model_acc()
            matrix_two()    
            print('======================================')
            print('======matrix generation complete======')
            print('======================================')
        except:
            get_modal_data()
            get_one_model_acc()
            get_two_model_acc()
            matrix_two()    
            print('======================================')
            print('======matrix generation complete======')
            print('======================================')    
    elif sys.argv[1]=="kernel":
        main_kernel()
        Result_analysis_kernel()
        print('=============================================================================')
        print('======The analysis of the impact of kernel size on accuracy is complete======')
        print('=============================================================================')
    elif sys.argv[1]=="all":
        try:
            Denoising_read()
            Result_analysis_4()
            print('===============================================')
            print('======FIGURE 4 matrix generation complete======')
            print('===============================================')
        except:
            Denoising_main()
            Denoising_read()
            Result_analysis_4()
            print('===============================================')
            print('======FIGURE 4 matrix generation complete======')
            print('===============================================')
        try:
            Result_analysis_6()
            print('===============================================')
            print('======FIGURE 6 matrix generation complete======')
            print('===============================================')
        except:
            Denoising_main()
            Denoising_read()
            Result_analysis_6()
            print('===============================================')
            print('======FIGURE 6 matrix generation complete======')
            print('===============================================')
            
  
        try:
            get_one_model_acc()
            get_two_model_acc()
            matrix_two()    
            print('======================================')
            print('======matrix generation complete======')
            print('======================================')
        except:
            get_modal_data()
            get_one_model_acc()
            get_two_model_acc()
            matrix_two()    
            print('======================================')
            print('======matrix generation complete======')
            print('======================================')    
        main_kernel()
        Result_analysis_kernel()
        print('=============================================================================')
        print('======The analysis of the impact of kernel size on accuracy is complete======')
        print('=============================================================================')