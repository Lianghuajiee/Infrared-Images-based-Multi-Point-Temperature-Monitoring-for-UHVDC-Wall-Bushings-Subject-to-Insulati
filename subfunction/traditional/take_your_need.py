import numpy as np
import cv2 as cv
from scipy.spatial import ConvexHull


def get_gradient_and_direction(image):
    """ Compute gradients and its direction
    Use Sobel filter to compute gradients and direction.
         -1 0 1        -1 -2 -1
    Gx = -2 0 2   Gy =  0  0  0
         -1 0 1         1  2  1

    Args:
        image: array of grey image

    Returns:
        gradients: the gradients of each pixel
        direction: the direction of the gradients of each pixel
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    W, H = image.shape
    gradients = np.zeros([W - 2, H - 2])
    direction = np.zeros([W - 2, H - 2])

    for i in range(W - 2):
        for j in range(H - 2):
            dx = np.sum(image[i:i+3, j:j+3] * Gx)
            dy = np.sum(image[i:i+3, j:j+3] * Gy)
            gradients[i, j] = np.sqrt(dx ** 2 + dy ** 2)
            if dx == 0:
                direction[i, j] = np.pi / 2
            else:
                direction[i, j] = np.arctan(dy / dx)

    gradients = np.uint8(gradients)
    return gradients, direction

def NMS(gradients, direction):
    """ Non-maxima suppression
    Args:
        gradients: the gradients of each pixel
        direction: the direction of the gradients of each pixel

    Returns:
        the output image
    """
    W, H = gradients.shape
    nms = np.copy(gradients[1:-1, 1:-1])

    for i in range(1, W - 1):
        for j in range(1, H - 1):
            theta = direction[i, j]
            weight = np.tan(theta)
            if theta > np.pi / 4:
                d1 = [0, 1]
                d2 = [1, 1]
                weight = 1 / weight
            elif theta >= 0:
                d1 = [1, 0]
                d2 = [1, 1]
            elif theta >= - np.pi / 4:
                d1 = [1, 0]
                d2 = [1, -1]
                weight *= -1
            else:
                d1 = [0, -1]
                d2 = [1, -1]
                weight = -1 / weight

            g1 = gradients[i + d1[0], j + d1[1]]
            g2 = gradients[i + d2[0], j + d2[1]]
            g3 = gradients[i - d1[0], j - d1[1]]
            g4 = gradients[i - d2[0], j - d2[1]]

            grade_count1 = g1 * weight + g2 * (1 - weight)
            grade_count2 = g3 * weight + g4 * (1 - weight)

            if grade_count1 > gradients[i, j] or grade_count2 > gradients[i, j]:
                nms[i - 1, j - 1] = 0
    return nms

def double_ksize_z(nms, ksize_z1, ksize_z2):
    """ Double ksize_z
    Use two ksize_zs to compute the edge.

    Args:
        nms: the input image
        ksize_z1: the low ksize_z
        ksize_z2: the high ksize_z

    Returns:
        The binary image.
    """
    visited = np.zeros_like(nms)
    output_image = nms.copy()
    W, H = output_image.shape

    def dfs(i, j):
        if i >= W or i < 0 or j >= H or j < 0 or visited[i, j] == 1:
            return
        visited[i, j] = 1
        if output_image[i, j] > ksize_z1:
            output_image[i, j] = 255
            dfs(i-1, j-1)
            dfs(i-1, j)
            dfs(i-1, j+1)
            dfs(i, j-1)
            dfs(i, j+1)
            dfs(i+1, j-1)
            dfs(i+1, j)
            dfs(i+1, j+1)
        else:
            output_image[i, j] = 0

    for w in range(W):
        for h in range(H):
            if visited[w, h] == 1:
                continue
            if output_image[w, h] >= ksize_z2:
                dfs(w, h)
            elif output_image[w, h] <= ksize_z1:
                output_image[w, h] = 0
                visited[w, h] = 1

    for w in range(W):
        for h in range(H):
            if visited[w, h] == 0:
                output_image[w, h] = 0
    return output_image

def canny_Outline (input_image,ksize_z):
    gradients, direction = get_gradient_and_direction(input_image)
    nms = NMS(gradients, direction)
    output_image = double_ksize_z(nms, ksize_z-5, ksize_z )        # canny轮廓图 
    return output_image

def save_to_txt(param_dict,name):
    with open('Data/Generated_data/local_argument_'+name+'.txt', 'w') as file:
        for local, params in param_dict.items():
            file.write(f"{local}: {params}\n")
def read_from_txt(name):
    param_dict = {}
    with open('Data/Generated_data/local_argument_'+name+'.txt', 'r') as file:
        for line in file:
            split_index = line.find(':')  # Find the first colon
            local = line[:split_index].strip()
            params = line[split_index + 1:].strip()
            param_dict[local] = eval(params)  # Using eval to convert the string representation of the dictionary back to a dictionary
    return param_dict 
    
# 修补平行四边形
def Patch_the_parallelogram(Outcome_site):
    # 修补直流套管的矩形不是平行四边形
    Outcome_mind_site = []
    Outcome_mind_site.append([Outcome_site[0,0],Outcome_site[0,1],Outcome_site[2,0],Outcome_site[2,1]])
    Outcome_mind_site.append([Outcome_site[1,0],Outcome_site[1,1],Outcome_site[3,0],Outcome_site[3,1]]) 
    up_K = (Outcome_mind_site[0][3] - Outcome_mind_site[0][1]) / (Outcome_mind_site[0][2]-Outcome_mind_site[0][0])
    up_B = Outcome_mind_site[0][1] - Outcome_mind_site[0][0] * up_K
    down_K = (Outcome_mind_site[1][3] - Outcome_mind_site[1][1]) / (Outcome_mind_site[1][2]-Outcome_mind_site[1][0])
    down_B = Outcome_mind_site[1][1] - Outcome_mind_site[1][0] * down_K
    
    mid_k = (up_K+down_K)/2

    up_l , up_r     = Outcome_mind_site[0][0] + mid_k*Outcome_mind_site[0][1] , Outcome_mind_site[0][2] + mid_k*Outcome_mind_site[0][3]
    down_l , down_r = Outcome_mind_site[1][0] + mid_k*Outcome_mind_site[1][1] , Outcome_mind_site[1][2] + mid_k*Outcome_mind_site[1][3]
    if up_l < down_l :
        K_orthogo = -1/up_K
        B_orthogo = Outcome_mind_site[0][1] - Outcome_mind_site[0][0] * K_orthogo
        Outcome_mind_site[1][0]= int((B_orthogo - down_B)/(down_K - K_orthogo))
        Outcome_mind_site[1][1]= int(Outcome_mind_site[1][0]*down_K + down_B) 
        
    elif up_l > down_l :
        K_orthogo = -1/down_K
        B_orthogo = Outcome_mind_site[1][1] - Outcome_mind_site[1][0] * K_orthogo
        Outcome_mind_site[0][0]= int((B_orthogo - up_B)/(up_K - K_orthogo))
        Outcome_mind_site[0][1]= int(Outcome_mind_site[0][0]*up_K + up_B) 
    
    if up_r < down_r :
        K_orthogo = -1/down_K
        B_orthogo = Outcome_mind_site[1][3] - Outcome_mind_site[1][2] * K_orthogo
        Outcome_mind_site[0][2]= int((B_orthogo - up_B)/(up_K - K_orthogo))
        Outcome_mind_site[0][3]= int(Outcome_mind_site[0][2]*up_K + up_B) 
        
    elif up_r > down_r :
        K_orthogo = -1/up_K
        B_orthogo = Outcome_mind_site[0][3] - Outcome_mind_site[0][2] * K_orthogo
        Outcome_mind_site[1][2]= int((B_orthogo - down_B)/(down_K - K_orthogo))
        Outcome_mind_site[1][3]= int(Outcome_mind_site[1][2]*down_K + down_B) 
    Outcome_site = np.array(Outcome_mind_site).reshape([-1,2])
    return Outcome_site 


def cut_image_conter(Outcome_site,Casing_profile,dst_back_gray_all,dst_back_rgb,image):
    y_con = int(np.mean(Outcome_site[:,0]))
    x_con =  int(np.mean(Outcome_site[:,1]))
    max_ri = 0 
 
    
    # 执行操作：减去474和164.5，然后取绝对值
    result = np.abs(Outcome_site - np.array([y_con, x_con]))
    # 找到绝对值最大的数
    max_value = int(np.max(result))
    
    piont=[]
    if y_con-max_value-10 >=0 :
        piont.append(y_con-max_value-10)   
        y_piont = y_con-max_value-10
    else:
        piont.append(0)
        y_piont = (y_con-max_value-10)//2
    if y_con+max_value+10 <= dst_back_gray_all.shape[1] :
        piont.append(y_con+max_value+10)   
    else:
        piont.append(dst_back_gray_all.shape[1])
        
    if x_con-max_value-10 >=0 :
        piont.append(x_con-max_value-10)   
        x_piont = x_con-max_value-10
    else:
        piont.append(0)
        x_piont = (x_con-max_value-10)//2
    if x_con+max_value+10 <= dst_back_gray_all.shape[0] :
        piont.append(x_con+max_value+10)   
    else:
        piont.append(dst_back_gray_all.shape[0])    
    
    
    
    square_rgb_image = np.zeros((max_value*2+20,max_value*2+20,3), dtype=np.uint8) 
    square_rgb_image[(square_rgb_image.shape[0]-(piont[3]-piont[2]))//2:(square_rgb_image.shape[0]-(piont[3]-piont[2]))//2+(piont[3]-piont[2]), (square_rgb_image.shape[0]-(piont[1]-piont[0]))//2:(square_rgb_image.shape[0]-(piont[1]-piont[0]))//2+(piont[1]-piont[0]),] = dst_back_rgb[ piont[2]:piont[3],piont[0]:piont[1],]
    dst_back_rgb = square_rgb_image
    
    square_rgb_image = np.zeros((max_value*2+20,max_value*2+20), dtype=np.uint8) 
    square_rgb_image[(square_rgb_image.shape[0]-(piont[3]-piont[2]))//2:(square_rgb_image.shape[0]-(piont[3]-piont[2]))//2+(piont[3]-piont[2]), (square_rgb_image.shape[0]-(piont[1]-piont[0]))//2:(square_rgb_image.shape[0]-(piont[1]-piont[0]))//2+(piont[1]-piont[0])] = dst_back_gray_all[ piont[2]:piont[3],piont[0]:piont[1]]
    dst_back_gray_all = square_rgb_image
    
    square_rgb_image = np.zeros((max_value*2+20,max_value*2+20,3), dtype=np.uint8) 
    square_rgb_image[(square_rgb_image.shape[0]-(piont[3]-piont[2]))//2:(square_rgb_image.shape[0]-(piont[3]-piont[2]))//2+(piont[3]-piont[2]), (square_rgb_image.shape[0]-(piont[1]-piont[0]))//2:(square_rgb_image.shape[0]-(piont[1]-piont[0]))//2+(piont[1]-piont[0]),] = image[ piont[2]:piont[3],piont[0]:piont[1],]
    image = square_rgb_image
    
    Casing_profile = Casing_profile - np.array([y_piont, x_piont])
    Outcome_site = Outcome_site - np.array([y_piont, x_piont])
    
    return Outcome_site,Casing_profile,dst_back_gray_all,dst_back_rgb ,image

class piont_take_cannula :
    def _init_():
        piont_take_cannula.save_argument_to_txt()
    def save_argument_to_txt():    
        d = {
             'local_00_0 (1)':  {'Casing_profile': [[285,237],[290,254],[360,217],[364,237]]}, #
             'local_00_0 (4)':  {'Casing_profile': [[294,242],[303,256],[374,222],[377,239]]}, #
             'local_00_0 (5)':  {'Casing_profile': [[298,237],[300,256],[386,215],[394,232]]}, 
             'local_01_1 (1)':  {'Casing_profile': [[200,285],[203,309],[412,160],[421,178]]}, 
             'local_01_1 (2)':  {'Casing_profile': [[192,349],[202,368],[417,227],[427,245]]}, 
             'local_01_1 (3)':  {'Casing_profile': [[188,351],[195,370],[413,235],[422,250]]}, 
             'local_02_0 (1)':  {'Casing_profile': [[275,207],[278,226],[358,186],[362,207]]}, 
             'local_02_0 (4)':  {'Casing_profile': [[302,313],[305,325],[362,299],[364,314]]}, 
             'local_02_0 (5)':  {'Casing_profile': [[289,310],[291,325],[352,298],[355,312]]}, 
             'local_02_0 (8)':  {'Casing_profile': [[270,282],[273,294],[327,267],[332,282]]}, 
             'local_02_1 (1)':  {'Casing_profile': [[278,256],[278,269],[336,244],[339,255]]},  
             'local_02_1 (2)':  {'Casing_profile': [[305,317],[308,331],[370,305],[371,319]]}, 
             'local_03_1 (1)':  {'Casing_profile': [[278,245],[270,262],[340,270],[334,286]]}, 
             'local_03_1 (2)':  {'Casing_profile': [[310,197],[304,213],[370,225],[362,239]]}, 
             'local_03_1 (3)':  {'Casing_profile': [[313,193],[306,208],[371,220],[364,236]]}, 
             'local_03_0 (1)':  {'Casing_profile': [[308,230],[299,252],[385,262],[377,288]]}, #
             'local_03_0 (4)':  {'Casing_profile': [[282,243],[279,260],[344,267],[336,283]]}, 
             'local_03_0 (3)':  {'Casing_profile': [[272,242],[264,259],[331,267],[324,281]]}, 
             'local_03_0 (5)':  {'Casing_profile': [[310,243],[302,259],[364,266],[358,281]]}, 
             'local_03_0 (7)':  {'Casing_profile': [[284,210],[275,234],[362,244],[350,262]]}, 
             'local_03_0 (8)':  {'Casing_profile': [[302,237],[293,258],[385,271],[377,289]]}, 
             'local_04_0 (11)': {'Casing_profile': [[238,247],[261,266],[282,192],[309,213]]}, 
             'local_04_0 (12)': {'Casing_profile': [[243,268],[263,278],[286,211],[311,225]]}, 
             'local_04_0 (13)': {'Casing_profile': [[242,259],[260,275],[280,213],[305,230]]}, 
             'local_04_1 (1)':  {'Casing_profile': [[227,285],[242,298],[260,247],[276,260]]}, 
             'local_04_1 (2)':  {'Casing_profile': [[239,259],[256,276],[283,202],[304,226]]}, 
             'local_04_1 (4)':  {'Casing_profile': [[247,300],[259,311],[280,262],[296,277]]}, 
             'local_04_1 (5)':  {'Casing_profile': [[256,298],[269,310],[289,262],[306,276]]},
             'local_04_1 (6)':  {'Casing_profile': [[268,299],[282,311],[302,260],[317,278]]},
             'local_04_1 (7)':  {'Casing_profile': [[255,298],[268,312],[287,262],[301,279]]},
             'local_04_1 (8)':  {'Casing_profile': [[247,269],[263,282],[280,230],[295,243]]},
             'local_05_1 (1)':  {'Casing_profile': [[356, 97],[353,110],[520,152],[516,166]]}, 
             'local_05_1 (2)':  {'Casing_profile': [[336,171],[332,183],[500,222],[495,235]]}, 
             'local_05_0 (1)':  {'Casing_profile': [[354,179],[352,190],[522,230],[518,242]]}, 
             'local_05_0 (2)':  {'Casing_profile': [[326,161],[324,173],[494,216],[490,227]]}, 
             'local_05_0 (8)':  {'Casing_profile': [[342,108],[337,120],[506,164],[504,176]]}, 
             'local_06_0 (6)':  {'Casing_profile': [[157,342],[166,358],[360,216],[370,231]]},
             'local_06_1 (1)':  {'Casing_profile': [[178,407],[186,425],[386,288],[396,304]]},
             'local_06_1 (2)':  {'Casing_profile': [[154,342],[163,359],[349,220],[360,237]]},  
             'local_07_1 (1)':  {'Casing_profile': [[199,118],[188,146],[571,293],[558,323]]}, 
             'local_07_1 (3)':  {'Casing_profile': [[230, 56],[217, 87],[595,230],[589,265]]}, 
             'local_08_1 (1)':  {'Casing_profile': [[144,451],[167,465],[325,210],[349,228]]}, 
             'local_08_1 (3)':  {'Casing_profile': [[172,464],[198,472],[341,265],[385,280]]}, 
             'local_09_0 (3)':  {'Casing_profile': [[ 96,453],[119,478],[321,158],[351,187]]}, 
             'local_10_0 (8)':  {'Casing_profile': [[220,230],[212,251],[494,340],[490,363]]}, 
             'local_10_0 (9)':  {'Casing_profile': [[257,230],[250,249],[532,339],[528,360]]}, 
             'local_10_0 (5)':  {'Casing_profile': [[273,236],[265,251],[475,316],[470,329]]}, 
             'local_10_1 (2)':  {'Casing_profile': [[255,184],[249,200],[457,266],[450,282]]}, 
             'local_10_1 (5)':  {'Casing_profile': [[272,236],[265,251],[476,317],[471,330]]}, 
             'local_11_0 (2)':  {'Casing_profile': [[252,313],[259,328],[306,291],[316,308]]}, 
             'local_11_1 (4)':  {'Casing_profile': [[255,341],[263,357],[312,321],[319,339]]}, 
             'local_11_1 (2)':  {'Casing_profile': [[249,340],[256,355],[310,319],[315,336]]}, 
             'local_11_1 (5)':  {'Casing_profile': [[252,341],[259,356],[316,318],[320,336]]}, 
             'local_11_1 (6)':  {'Casing_profile': [[255,344],[263,359],[319,322],[322,340]]}, 
             'local_12_0 (1)':  {'Casing_profile': [[379,263],[369,283],[468,284],[464,305]]}, 
             'local_12_0 (4)':  {'Casing_profile': [[346,260],[344,277],[407,275],[405,290]]}, 
             'local_12_0 (7)':  {'Casing_profile': [[376,245],[370,264],[460,268],[456,285]]}, 
             'local_12_0 (8)':  {'Casing_profile': [[350,267],[346,283],[437,288],[435,306]]}, 
             'local_12_0 (9)':  {'Casing_profile': [[380,262],[372,284],[470,284],[465,303]]},
             'local_12_0 (10)': {'Casing_profile': [[343,266],[341,282],[406,279],[404,296]]}, 
             'local_12_0 (11)': {'Casing_profile': [[407,261],[396,283],[491,282],[489,304]]}, 
             'local_12_1 (2)':  {'Casing_profile': [[346,209],[340,222],[409,225],[406,241]]},
             'local_13_0 (1)':  {'Casing_profile': [[351,308],[368,330],[594,145],[595,179]]},
             'local_13_1 (2)':  {'Casing_profile': [[274,321],[286,337],[459,197],[471,212]]}, 
             'local_13_1 (4)':  {'Casing_profile': [[316,326],[326,343],[499,199],[511,215]]}, 
             'local_14_1 (1)':  {'Casing_profile': [[266,237],[257,251],[422,340],[420,360]]},  
             'local_14_1 (3)':  {'Casing_profile': [[257,233],[246,248],[418,344],[413,365]]}, 
             'local_14_1 (5)':  {'Casing_profile': [[260,183],[243,207],[466,327],[462,356]]}, 
             'local_15_1 (5)':  {'Casing_profile': [[342,  3],[294, 39],[595,229],[596,298]]},  
             'local_15_1 (1)':  {'Casing_profile': [[313, 54],[270, 85],[587,319],[562,358]]}, #
             'local_15_0 (9)':  {'Casing_profile': [[255, 76],[230,103],[588,371],[570,399]]}, 
                        
            }
        save_to_txt(d,'piont') 
    def get_local_argument(path):   
    
        argument_all=read_from_txt('piont')
        argument_path=argument_all[path]
        Centre = np.array(argument_path['Casing_profile'])
      
        return Centre
        
    def work (name):
        Outcome_site = piont_take_cannula.get_local_argument(name)
        Outcome_site = np.array(Outcome_site)

        
        Outcome_site = Patch_the_parallelogram(Outcome_site)
        return Outcome_site

def Laplace(img, kernel):

    des_8U = cv.filter2D(img, -1, kernel=kernel, borderType=cv.BORDER_DEFAULT)
    des_16S = cv.filter2D(img, ddepth=cv.CV_16SC1, kernel=kernel, borderType=cv.BORDER_DEFAULT)

    g = img - des_16S
    g[g<0] = 0
    g[g>255] = 255
    g = np.uint8(g)
    return g    
def smooth(image, sigma = 1.4, length = 5):
    """ Smooth the image
    Compute a gaussian filter with sigma = sigma and kernal_length = length.
    Each element in the kernal can be computed as below:
        G[i, j] = (1/(2*pi*sigma**2))*exp(-((i-k-1)**2 + (j-k-1)**2)/2*sigma**2)
    Then, use the gaussian filter to smooth the input image.

    Args:
        image: array of grey image
        sigma: the sigma of gaussian filter, default to be 1.4
        length: the kernal length, default to be 5

    Returns:
        the smoothed image
    """
    # Compute gaussian filter
    k = length // 2
    gaussian = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            gaussian[i, j] = np.exp(-((i-k) ** 2 + (j-k) ** 2) / (2 * sigma ** 2))
    gaussian /= 2 * np.pi * sigma ** 2
    # Batch Normalization
    gaussian = gaussian / np.sum(gaussian)

    # Use Gaussian Filter
    W, H = image.shape
    new_image = np.zeros([W - k * 2, H - k * 2])

    for i in range(W - 2 * k):
        for j in range(H - 2 * k):
            # 卷积运算
            new_image[i, j] = np.sum(image[i:i+length, j:j+length] * gaussian)

    new_image = np.uint8(new_image)
    return new_image    
    
def find_line_take_cannula_integration(name,image,img0):
        kernel1 = np.asarray([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]])
        kernel2 = np.asarray([[0, 1, 0],  
                              [1, -4, 1],
                              [0, 1, 0]])  
        Centre,angle_length,ksize_z,HoughLines_or_LSD,range_number,smooth_size,sigma,threshold= find_line_take_cannula.get_local_argument(name)
    
        image = image[smooth_size:,smooth_size:]
        smoothed_image = smooth(img0, sigma = sigma , length = smooth_size)                       # 高斯去噪

        img0 = img0[smooth_size:,smooth_size:]
        output_image = canny_Outline(smoothed_image,threshold)
         
        output_image = Laplace(output_image, kernel2)  
        
        output_image,Outcome_site=find_line_take_cannula.LSD_Designated_area(output_image,Centre,angle_length,ksize_z,HoughLines_or_LSD,0.97,range_number)
    
        return output_image,Outcome_site,image,img0    
    
       
    
class find_line_take_cannula :
    
    def save_argument_to_txt():     # _0 白天     _1 黑夜
        d = {'local_00_0': {'Centre': [[377,147],[282,279]], 'angle_length': -16.6 , 'HoughLines_or_LSD': 0 , 'range_number' : 1 , 'ksize_z' : 3 , 'smooth_size': 7 , 'sigma' : 1.4 , 'threshold' : 28 },
             'local_01_0': {'Centre': [[406, 71],[211,380]], 'angle_length': -28.1 , 'HoughLines_or_LSD': 0 , 'range_number' : 2 , 'ksize_z' : 15 , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 20 }, 
             'local_02_0': {'Centre': [[347,164],[322,323]], 'angle_length': -10.5 , 'HoughLines_or_LSD': 1 , 'range_number' : 1 , 'ksize_z' : 3 , 'smooth_size': 7  , 'sigma' : 1.4 , 'threshold' : 18 },
             'local_03_0': {'Centre': [[286,172],[335,322]], 'angle_length':  19.7 , 'HoughLines_or_LSD': 1 , 'range_number' : 2 , 'ksize_z' : 10 , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 20 },
             'local_04_0': {'Centre': [[246,114],[279,352]], 'angle_length': -45.0 , 'HoughLines_or_LSD': 0 , 'range_number' : 2 , 'ksize_z' : 10 , 'smooth_size': 7  , 'sigma' : 1.4 , 'threshold' : 20 },
             'local_05_0': {'Centre': [[412,113],[477,242]], 'angle_length':  17.5 , 'HoughLines_or_LSD': 0 , 'range_number' : 3 , 'ksize_z' : 12 , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 20 },
             'local_06_0': {'Centre': [[397,144],[200,429]], 'angle_length': -30.2 , 'HoughLines_or_LSD': 1 , 'range_number' : 2 , 'ksize_z' : 10 , 'smooth_size': 7  , 'sigma' : 1.4 , 'threshold' : 13 },
             'local_07_0': {'Centre': [[238, 69],[492,326]], 'angle_length':  22.5 , 'HoughLines_or_LSD': 1 , 'range_number' : 1 , 'ksize_z' : 3 , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_08_0': {'Centre': [[285,243],[216,478]], 'angle_length': -50.0 , 'HoughLines_or_LSD': 1 , 'range_number' : 3 , 'ksize_z' : 10 , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_09_0': {'Centre': [[229,464],[276,163]], 'angle_length': -52.0 , 'HoughLines_or_LSD': 1 , 'range_number' : 1 , 'ksize_z' : 3 , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_10_0': {'Centre': [[479,376],[255,186]], 'angle_length':  21.3 , 'HoughLines_or_LSD': 1 , 'range_number' : 2 , 'ksize_z' : 15 , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_11_0': {'Centre': [[268,338],[309,213]], 'angle_length': -20.5 , 'HoughLines_or_LSD': 1 , 'range_number' : 1 , 'ksize_z' : 3  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 20 },
             'local_12_0': {'Centre': [[385,229],[398,304]], 'angle_length':  15.7 , 'HoughLines_or_LSD': 1 , 'range_number' : 2 , 'ksize_z' : 10  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 25 },
             'local_13_0': {'Centre': [[444,128],[384,337]], 'angle_length': -34.5 , 'HoughLines_or_LSD': 1 , 'range_number' : 2 , 'ksize_z' : 10  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },  
             'local_14_0': {'Centre': [[303,138],[409,429]], 'angle_length':  35.0 , 'HoughLines_or_LSD': 1 , 'range_number' : 1 , 'ksize_z' : 3  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_15_0': {'Centre': [[263, 74],[553,473]], 'angle_length':  43.6 , 'HoughLines_or_LSD': 1 , 'range_number' : 1 , 'ksize_z' : 3  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             
             'local_00_1': {'Centre': [[305,230],[271,396]], 'angle_length': -16.6 , 'HoughLines_or_LSD': 1 , 'range_number' : 1 , 'ksize_z' : 3  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_01_1': {'Centre': [[395,154],[218,375]], 'angle_length': -28.1 , 'HoughLines_or_LSD': 0 , 'range_number' : 1 , 'ksize_z' : 5  , 'smooth_size': 7  , 'sigma' : 1.4 , 'threshold' : 15 }, 
             'local_02_1': {'Centre': [[321,341],[329,234]], 'angle_length': -10.5 , 'HoughLines_or_LSD': 0 , 'range_number' : 1 , 'ksize_z' : 3  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_03_1': {'Centre': [[328,297],[315,180]], 'angle_length':  19.7 , 'HoughLines_or_LSD': 0 , 'range_number' : 1 , 'ksize_z' : 3  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 20 },
             'local_04_1': {'Centre': [[278,333],[255,206]], 'angle_length': -45.0 , 'HoughLines_or_LSD': 1 , 'range_number' : 1 , 'ksize_z' : 3  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 28 },
             'local_05_1': {'Centre': [[406,104],[506,252]], 'angle_length':  17.5 , 'HoughLines_or_LSD': 1 , 'range_number' : 1 , 'ksize_z' : 3  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_06_1': {'Centre': [[190,434],[342,207]], 'angle_length': -30.2 , 'HoughLines_or_LSD': 0 , 'range_number' : 2 , 'ksize_z' : 5  , 'smooth_size': 7  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_07_1': {'Centre': [[290, 71],[473,310]], 'angle_length':  22.5 , 'HoughLines_or_LSD': 0 , 'range_number' : 2 , 'ksize_z' : 25  , 'smooth_size': 7  , 'sigma' : 1.4 , 'threshold' : 17 },
             'local_08_1': {'Centre': [[300,188],[223,473]], 'angle_length': -46.0 , 'HoughLines_or_LSD': 0 , 'range_number' : 3 , 'ksize_z' : 20  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 10 },
             'local_09_1': {'Centre': [[229,464],[276,163]], 'angle_length': -52.0 , 'HoughLines_or_LSD': 1 , 'range_number' : 1 , 'ksize_z' : 3 , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_10_1': {'Centre': [[476,371],[257,160]], 'angle_length':  21.3 , 'HoughLines_or_LSD': 1 , 'range_number' : 2 , 'ksize_z' : 10  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_11_1': {'Centre': [[265,367],[293,259]], 'angle_length': -20.5 , 'HoughLines_or_LSD': 1 , 'range_number' : 1 , 'ksize_z' : 3  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_12_1': {'Centre': [[348,196],[404,254]], 'angle_length':  15.7 , 'HoughLines_or_LSD': 0 , 'range_number' : 1 , 'ksize_z' : 1  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_13_1': {'Centre': [[520,135],[315,355]], 'angle_length': -34.5 , 'HoughLines_or_LSD': 1 , 'range_number' : 2 , 'ksize_z' : 10  , 'smooth_size': 7  , 'sigma' : 1.4 , 'threshold' : 15 },  
             'local_14_1': {'Centre': [[303,138],[418,365]], 'angle_length':  35.0 , 'HoughLines_or_LSD': 0 , 'range_number' : 2 , 'ksize_z' : 10  , 'smooth_size': 6  , 'sigma' : 1.4 , 'threshold' : 15 },
             'local_15_1': {'Centre': [[355,  1],[630,448]], 'angle_length':  40.9 , 'HoughLines_or_LSD': 0 , 'range_number' : 5 , 'ksize_z' : 20  , 'smooth_size': 8  , 'sigma' : 3.4 , 'threshold' : 20 } #
             
            }
            
  
        save_to_txt(d,'line') 
            
    def get_local_argument(path):   
    
        argument_all=read_from_txt('line')
        argument_path=argument_all[path]
        
        Centre = np.array(argument_path['Centre'])
        angle_length = argument_path['angle_length']
        HoughLines_or_LSD = argument_path['HoughLines_or_LSD']
        range_number = argument_path['range_number']
        ksize_z = argument_path['ksize_z']
        smooth_size = argument_path['smooth_size']
        sigma = argument_path['sigma']
        threshold = argument_path['threshold']
        return Centre,angle_length,ksize_z,HoughLines_or_LSD,range_number,smooth_size,sigma,threshold
    def Convex_Hull (points):  #凸包顶点
        hull = ConvexHull(points)

        return points[hull.vertices]
    def point_in_convex_hull_cross_product(p, convex_hull_point):
        for i in range(convex_hull_point.shape[0]):
            p1 = convex_hull_point[i]
            p2 = convex_hull_point[(i + 1) % len(convex_hull_point)]
            if (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) < 0:
                return False
        return True
    
    def Edge_connection(length_site,angle):       
        
        length_site=np.array(length_site)           #去除偏离太多的线段
        tt=[]       # 标记角度偏离过大的线段位置
        K_B_longer =[]      # 记录线段的K B 长度 数值
        length_angle=[]     # 记录线段的angle数值
        for i in range(length_site.shape[0]): 
            x01=length_site[i,0]
            y01=length_site[i,1]
            x02=length_site[i,2]
            y02=length_site[i,3]
            if x01 == x02 :
                k1 = 100
            else :
                k1 = (y01 - y02) / (x01-x02)
            b =  y01 - k1 * x01
            longer =  (y01 - y02)**2 + (x01-x02)**2   
            K_B_longer.append([k1 , b , longer])
            length_angle.append( np.arctan(k1)*180/np.pi )
            if not angle - 5 <= length_angle[-1] <=angle + 5 :
                tt.append(i)
        
        tt.sort()
        tt.reverse()
        K_B_longer = np.array( K_B_longer)
        if len(tt) != 0 :
            for o in tt:
                length_site=np.delete(length_site, o, axis=0)
                K_B_longer=np.delete(K_B_longer, o, axis=0)
                del length_angle[o]
        
        length_site = np.array(length_site)
        
        
        
        
        
        
        
        center = np.mean(length_site, axis=0)
        
        # center_liat2 = length_site.reshape(-1,2)                      #理论使用凸包顶点算中心更好，实际不行 
        # center_liat2 = find_line_take_cannula.Convex_Hull(center_liat2)
        # length_site1 = []
        # length_site1.append(center_liat2[1])
        # length_site1.append(center_liat2[2])
        # length_site1.append(center_liat2[0])
        # length_site1.append(center_liat2[3])
        # length_site1 = np.array(length_site1)
        # length_site1 = length_site1.reshape(-1,4)
        # center = np.mean(length_site1, axis=0)
        
        
        
        
        angle_mean = np.mean(length_angle)
        
        mid_k = np.tan(angle_mean*np.pi/180)
        mid_b = (center[1]+center[3])/2 - (center[0]+center[2])/2 * mid_k
        up_length_piont = []
        up_K_B_longer = []
        down_length_piont = []
        down_K_B_longer = []
        for i , itme in enumerate (length_site):
            if  mid_b > itme[1] - itme[0] * mid_k :
                up_length_piont.append([itme[0],itme[1]])
                up_length_piont.append([itme[2],itme[3]])
                up_K_B_longer.append( K_B_longer [i] )
            else: 
                down_length_piont.append([itme[0],itme[1]])
                down_length_piont.append([itme[2],itme[3]])
                down_K_B_longer.append( K_B_longer [i] )
        
        up_length_piont=np.array(up_length_piont)
        down_length_piont=np.array(down_length_piont)
        up_K_B_longer=np.array(up_K_B_longer)
        down_K_B_longer=np.array(down_K_B_longer)

        up_length_piont = up_length_piont[up_length_piont[:,0].argsort()] #按照第1列对行排序
        down_length_piont = down_length_piont[down_length_piont[:,0].argsort()] #按照第1列对行排序
        up_K_B_longer = up_K_B_longer[up_K_B_longer[:,2].argsort()] #按照第3列对行排序
        down_K_B_longer = down_K_B_longer[down_K_B_longer[:,2].argsort()] #按照第3列对行排序
        

            
        up_l_y=int(up_length_piont[0,0]*up_K_B_longer[-1,0] + up_K_B_longer[-1,1])
        up_r_y=int(up_length_piont[-1,0]*up_K_B_longer[-1,0] + up_K_B_longer[-1,1])
        
        down_l_y=int(down_length_piont[0,0]*down_K_B_longer[-1,0] + down_K_B_longer[-1,1])
        down_r_y=int(down_length_piont[-1,0]*down_K_B_longer[-1,0] + down_K_B_longer[-1,1])
        
        Outcome_mind_site=[]
        Outcome_mind_site.append([up_length_piont[0,0],up_l_y,up_length_piont[-1,0],up_r_y])
        Outcome_mind_site.append([down_length_piont[0,0],down_l_y,down_length_piont[-1,0],down_r_y])   


        # 修补直流套管的矩形不是平行四边形
        up_K = (Outcome_mind_site[0][3] - Outcome_mind_site[0][1]) / (Outcome_mind_site[0][2]-Outcome_mind_site[0][0])
        up_B = Outcome_mind_site[0][1] - Outcome_mind_site[0][0] * up_K
        down_K = (Outcome_mind_site[1][3] - Outcome_mind_site[1][1]) / (Outcome_mind_site[1][2]-Outcome_mind_site[1][0])
        down_B = Outcome_mind_site[1][1] - Outcome_mind_site[1][0] * down_K
        
        mid_k = (up_K+down_K)/2

        up_l , up_r     = Outcome_mind_site[0][0] + mid_k*Outcome_mind_site[0][1] , Outcome_mind_site[0][2] + mid_k*Outcome_mind_site[0][3]
        down_l , down_r = Outcome_mind_site[1][0] + mid_k*Outcome_mind_site[1][1] , Outcome_mind_site[1][2] + mid_k*Outcome_mind_site[1][3]
        if up_l < down_l :
            K_orthogo = -1/up_K
            B_orthogo = Outcome_mind_site[0][1] - Outcome_mind_site[0][0] * K_orthogo
            Outcome_mind_site[1][0]= int((B_orthogo - down_B)/(down_K - K_orthogo))
            Outcome_mind_site[1][1]= int(Outcome_mind_site[1][0]*down_K + down_B) 
            
        elif up_l > down_l :
            K_orthogo = -1/down_K
            B_orthogo = Outcome_mind_site[1][1] - Outcome_mind_site[1][0] * K_orthogo
            Outcome_mind_site[0][0]= int((B_orthogo - up_B)/(up_K - K_orthogo))
            Outcome_mind_site[0][1]= int(Outcome_mind_site[0][0]*up_K + up_B) 
        
        if up_r < down_r :
            K_orthogo = -1/down_K
            B_orthogo = Outcome_mind_site[1][3] - Outcome_mind_site[1][2] * K_orthogo
            Outcome_mind_site[0][2]= int((B_orthogo - up_B)/(up_K - K_orthogo))
            Outcome_mind_site[0][3]= int(Outcome_mind_site[0][2]*up_K + up_B) 
            
        elif up_r > down_r :
            K_orthogo = -1/up_K
            B_orthogo = Outcome_mind_site[0][3] - Outcome_mind_site[0][2] * K_orthogo
            Outcome_mind_site[1][2]= int((B_orthogo - down_B)/(down_K - K_orthogo))
            Outcome_mind_site[1][3]= int(Outcome_mind_site[1][2]*down_K + down_B) 

        return Outcome_mind_site
    def shrink_to_center(coordinates,shrink_ratio):
        # 遍历每个坐标点
        coordinates=np.array(coordinates)
        # 计算多边形的中心点坐标
        center = np.mean(coordinates, axis=0)
        
        for point in coordinates:
            
            # 计算顶点与中心点的偏移量
            offset =   point - center
            # 缩小顶点向中心点的程度
            # shrink_ratio = 0.5  # 缩小比例，可以根据需要进行调整
            scaled_offset = offset * shrink_ratio
            # 调整顶点位置
            new_point = center + scaled_offset
            # 更新坐标点为调整后的位置
            point[0] = new_point[0]
            point[1] = new_point[1]
        return coordinates       
    def LSD_Designated_area(img0,Centre,angle_length,Length2,HoughLines_or_LSD,size_close,range_number): # Length:是正方形边长1/2    Length2:检索过目标框线段附近的散落线段不连续线段
    
        # HoughLines_or_LSD = 0      # LSD and HoughLines:0  LSD:1   HoughLines:2 
        Outcome_mind_site=[]
        if  HoughLines_or_LSD == 0    :
            lsd = cv.createLineSegmentDetector(0)
            dlines = lsd.detect(img0)
            for dline in dlines[0]:     
                x0 = int(round(dline[0][0]))
                y0 = int(round(dline[0][1]))
                x1 = int(round(dline[0][2]))
                y1 = int(round(dline[0][3]))
                if x0 == x1 :
                    k = 100
                else :
                    k = (y0 - y1) / (x0-x1)
                b = y0 - x0 * k
                mind=[]
                for i in range(2):
                    mind.append(float(Centre[i,1]-k*Centre[i,0]-b))
                    
                if(max(mind)*min(mind)<0):          # 提取过目标框的直线
                    Outcome_mind_site.append([x0,y0,x1,y1])    #下次用字典方便查看 
                   
            lines = cv.HoughLinesP(img0,1,np.pi/180,50,minLineLength=20,maxLineGap=2)
            for line in lines:
                x0,y0,x1,y1 = line[0]
                if x0 == x1 :
                    k = 100
                else :
                    k = (y0 - y1) / (x0-x1)
                b = y0 - x0 * k
                mind=[]
                for i in range(2):
                    mind.append(float(Centre[i,1]-k*Centre[i,0]-b))
                if(max(mind)*min(mind)<0):          # 提取过目标框的直线
                    Outcome_mind_site.append([x0,y0,x1,y1])    #下次用字典方便查看        
                  
        elif HoughLines_or_LSD == 1 :
            lsd = cv.createLineSegmentDetector(0)
            dlines = lsd.detect(img0)
            for dline in dlines[0]:     
                x0 = int(round(dline[0][0]))
                y0 = int(round(dline[0][1]))
                x1 = int(round(dline[0][2]))
                y1 = int(round(dline[0][3]))
                if x0 == x1 :
                    k = 100
                else :
                    k = (y0 - y1) / (x0-x1)
                b = y0 - x0 * k
                mind=[]
                for i in range(2):
                    mind.append(float(Centre[i,1]-k*Centre[i,0]-b))
                if(max(mind)*min(mind)<0):          # 提取过目标框的直线
                    Outcome_mind_site.append([x0,y0,x1,y1])    #下次用字典方便查看       
                   
        elif HoughLines_or_LSD == 2 : 
            lines = cv.HoughLinesP(img0,1,np.pi/180,12,minLineLength=20,maxLineGap=6)
            for line in lines:
                x0,y0,x1,y1 = line[0]
                if x0 == x1 :
                    k = 100
                else :
                    k = (y0 - y1) / (x0-x1)
                b = y0 - x0 * k
                mind=[]
                for i in range(2):
                    mind.append(float(Centre[i,1]-k*Centre[i,0]-b))
                if(max(mind)*min(mind)<0):          # 提取过目标框的直线
                    Outcome_mind_site.append([x0,y0,x1,y1])    #下次用字典方便查看       
                   
       

        Outcome_site=[]    
        for dline in Outcome_mind_site:         # 提取过目标框的线段
            # if  (((Centre[0,0]-dline[0]) * (Centre[0,0]-dline[2]))<0 or ((Centre[1,0]-dline[0]) * (Centre[1,0]-dline[2]))<0) or (max(dline[0],dline[2])<max(Centre[1,0],Centre[0,0]) and min(dline[0],dline[2])>min(Centre[1,0],Centre[0,0])):

            if (((Centre[0,0]-dline[0]) * (Centre[1,0]-dline[0]))<0 and ((Centre[1,1]-dline[1]) * (Centre[0,1]-dline[1]))<0) or \
                (((Centre[0,0]-dline[2]) * (Centre[1,0]-dline[2]))<0 and ((Centre[1,1]-dline[3]) * (Centre[0,1]-dline[3]))<0) or \
                (max(dline[0],dline[2]) > max(Centre[0,0],Centre[1,0]) and min(dline[0],dline[2]) < min(Centre[0,0],Centre[1,0])):
                x01=dline[0]
                y01=dline[1]
                x02=dline[2]
                y02=dline[3]
                if x01 == x02 :
                    k1 = 100
                else :
                    k1 = (y01 - y02) / (x01-x02)
                if angle_length - 10 <= np.arctan(k1)*180/np.pi <=angle_length + 10:
                    Outcome_site.append(dline)
        for dline in Outcome_site:        
            Outcome_mind_site.remove(dline) 
        

        
        for _ in range(range_number):
            Outcome_mind3_site=find_line_take_cannula.Convex_Hull(np.array(Outcome_site).reshape([-1,2]))      #两点确定一个矩形，矩形加上Length2后提取含有的相交的线段
            Outcome_mind_site = np.array(Outcome_mind_site)
        
            for dline in Outcome_mind_site:         # 提取过目标框的线段附近的线段
                dline = np.array(dline)
                x01=dline[0]
                y01=dline[1]
                x02=dline[2]
                y02=dline[3]
                if x01 == x02 :
                    k1 = 100
                else :
                    k1 = (y01 - y02) / (x01-x02)
                if angle_length - 10 <= np.arctan(k1)*180/np.pi <=angle_length + 10:
                    if find_line_take_cannula.point_in_convex_hull_cross_product(dline[:2],Outcome_mind3_site) :
                        Outcome_site = np.concatenate((Outcome_site,[dline]),axis=0)
                    elif find_line_take_cannula.point_in_convex_hull_cross_product(dline[2:],Outcome_mind3_site) :   
                        Outcome_site = np.concatenate((Outcome_site,[dline]),axis=0)
                    else :
                        for hull_point in Outcome_mind3_site:
                            hull_point_1 = dline[:2] - hull_point
                            if (hull_point_1[0]**2 + hull_point_1[1]**2)<= Length2**2:
                                Outcome_site = np.concatenate((Outcome_site,[dline]),axis=0)
                            hull_point_1 = dline[2:] - hull_point
                            if (hull_point_1[0]**2 + hull_point_1[1]**2)<= Length2**2:
                                Outcome_site = np.concatenate((Outcome_site,[dline]),axis=0)        
                        
        Outcome_site = find_line_take_cannula.Edge_connection(Outcome_site,angle_length)    #线段处理
        Outcome_site = np.array(Outcome_site)
        Outcome_site = Outcome_site.reshape([-1,2])
        
        for ob in range(2):     #向中心靠拢
            Outcome_site[0+ob,1]+= 2
            Outcome_site[2+ob,1]-= 2
            
            
        Outcome_site = find_line_take_cannula.shrink_to_center(Outcome_site,size_close)
        img0 = cv.cvtColor(img0, cv.COLOR_GRAY2RGB)     #转为彩色
        for i in range(0,Outcome_site.shape[0],2):     
            cv.line(img0, (Outcome_site[i][0], Outcome_site[i][1]), (Outcome_site[i+1][0],Outcome_site[i+1][1]), (0,0,255), 1, cv.LINE_AA)
        cv.line(img0, (Centre[0,0], Centre[0,1]), (Centre[1,0], Centre[1,1]), (0,255,255), 1, cv.LINE_AA) 

        return  img0,Outcome_site    
        