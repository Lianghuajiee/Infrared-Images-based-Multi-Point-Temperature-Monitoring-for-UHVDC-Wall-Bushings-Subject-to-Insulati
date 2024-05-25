import numpy as np
import time
import math
from tqdm import tqdm
import sys


# 高斯降噪
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
    
    
 
class NLM:
    def __init__(self,dst_back_rgb):
        # #Load image
        # # self.img_path = "noisy_image2.jpg"
        # self.img_path = "cut.jpg"
        # image = IMG.open(self.img_path)

        # image = image.convert('L')#转灰度图
        self.image = np.array(dst_back_rgb) 

        self.results = []
        self.method_noise = []
        self.method_names = []

    def GaussianTemplate(self, kernel_size, sigma):
        """
        @description  : Given a gaussian distribution, this function can generate a kernel_size * kernel_size matrix. 
        @param  :  kernel_size = size of matrix
        @param  :  sigma = sigma of the gaussian distribution
        @Returns  : the matrix
        """
        

        template = np.ones(shape = (kernel_size, kernel_size), dtype="float64", )

        k = int(kernel_size / 2)
        k1 = 1/(2*np.pi*sigma*sigma)
        k2 = 2*sigma*sigma

        SUM = 0.0
        for i in range(kernel_size):
            for j in range(kernel_size):
                template[i,j] = k1*np.exp(-((i-k)*(i-k)+(j-k)*(j-k)) / k2)
                SUM += template[i,j]

        for i in range(kernel_size):
            for j in range(kernel_size):
                template[i,j] /= SUM
        #print("Gaussian Template = \n", template)
        return template

    def Gaussian_Filtering(self, src, dst = [], kernel_size = 3, sigma=0.8):
        """
        @description  : Given a image, return the gaussian filter result.
        @param  : src : source image
        @param  : dst: destination/ result image
        @param  : kernel_size: the conv kernel size.
        @param  : sigma: sigma of gaussian distribution
        @Returns  : Image that has been gaussian filtered
        """
        print("Gaussian Filtering start. Kernel size = {}, sigma = {}".format(kernel_size,sigma))
        start_time = time.time()
        if kernel_size == 1:
            dst = src
            return dst
        elif kernel_size%2 == 0 or kernel_size <= 0:
            print("卷积核大小必须是大于1的奇数")
            return -1

        padding_size = int((kernel_size - 1) / 2)
        img_width = src.shape[0] + padding_size*2
        img_height = src.shape[1] + padding_size*2

        tmp = np.zeros((img_width,img_height))
        dst = np.zeros((src.shape[0],src.shape[1]))
        #padding
        for i in range(padding_size,img_width-padding_size):
            for j in range(padding_size,img_height-padding_size):
                tmp[i,j] = src[i-padding_size,j-padding_size]
        kernel = self.GaussianTemplate(kernel_size, sigma)
        #Gaussian Filtering
        for row in range(padding_size, img_width-padding_size):
            for col in range(padding_size, img_height-padding_size):
                #sum = [0.0,0.0,0.0] #3Channel
                sum = 0
                for i in range(-padding_size,padding_size+1):
                    for j in range(-padding_size,padding_size+1):
                        #for channel in range(0,3):
                            #sum[channel] += tmp[(row+i),(col+j),channel] * kernel[i+padding_size,j+padding_size]
                        sum += tmp[(row+i),(col+j)] * kernel[i+padding_size,j+padding_size]
                if sum > 255:
                    sum = 255
                if sum < 0:
                    sum = 0
                dst[row - padding_size, col - padding_size] = sum
        dst = dst.astype('int32')
        end_time = time.time()

        print('Gaussian Filtering Complete. Time:{}'.format(end_time-start_time))
        method_noise = src - dst
        #Visualization
        self.results.append(dst)
        self.method_noise.append(method_noise)
        self.method_names.append("Gaussian Filtering")

        return dst, method_noise

    def Anisotropic_Filtering(self, src, dst=[], iterations = 10, k = 15, _lambda = 0.25):
        """
        @description  : Anisotropic filtering
        @param  : src : source image
        @param  : dst: destination/ result image
        @Returns  : Anisotropic filtering result
        """
        print("Anisotropic Filtering start. iterations = {}, k = {}, lambda = {}".format(iterations,k,_lambda))
        start_time = time.time()
        image_width = src.shape[0]
        image_height = src.shape[1]
        k2 = 1.0*k*k                                  # Since we only need k^2
        old_dst = src.copy().astype("float64")
        new_dst = src.copy().astype("float64")


        for i in range(iterations):
            for row in range(1,image_width-1):
                for col in range(1,image_height-1):
                    N_grad = old_dst[row-1,col] - old_dst[row,col]
                    S_grad = old_dst[row+1,col] - old_dst[row,col]
                    E_grad = old_dst[row,col-1] - old_dst[row,col]
                    W_grad = old_dst[row,col+1] - old_dst[row,col]
                    N_c = np.exp(-N_grad*N_grad/k2)
                    S_c = np.exp(-S_grad*S_grad/k2)
                    E_c = np.exp(-E_grad*E_grad/k2)
                    W_c = np.exp(-W_grad*W_grad/k2)
                    new_dst[row,col] = old_dst[row,col] + _lambda *(N_grad*N_c + S_grad*S_c + E_grad*E_c + W_grad*W_c)
            old_dst = new_dst

        dst = new_dst#.astype("uint8")
        end_time = time.time()
        print("Anisotropic filtering complete. Time:{}".format(end_time-start_time))
        method_noise = src - dst
        
        self.results.append(dst)
        self.method_noise.append(method_noise)
        self.method_names.append("Anisotropic Filtering")

        return dst,method_noise
    def Total_Variation_Minimization(self, src, dst=[], iterations = 100, _lambda = 0.03):
        """
        @description  : Total variation minimization
        @param  : src: source image
        @param  : dst: destination/ result image
        @Returns  : Total variation minimization result
        """
        print("Total Variation Minimization start. iterations = {}, lambda = {}".format(iterations,_lambda))
        start_time = time.time()
        image_width = src.shape[0]
        image_height = src.shape[1]
        dst = src.copy()
        u0 = src.copy()
        h = 1
        Energy = []
        cnt= 0
        for i in range(0,iterations):
            for row in range(1,image_width-1):
                for col in range(1,image_height-1):
                    ux = (float(dst[row + 1, col]) - float(dst[row, col])) / h
                    uy = (float(dst[row, col + 1]) - float(dst[row, col - 1])) / (2 * h)
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c1 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c1 = 1 / grad_u

                    ux = (float(dst[row, col]) - float(dst[row - 1, col])) / h
                    uy = (float(dst[row - 1, col + 1]) - float(dst[row - 1, col - 1])) / (2 * h)
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c2 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c2 = 1 / grad_u

                    ux = (float(dst[row + 1, col]) - float(dst[row - 1, col])) / (2 * h)
                    uy = (float(dst[row, col + 1]) - float(dst[row, col])) / h
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c3 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c3 = 1 / grad_u

                    ux = (float(dst[row + 1, col - 1]) - float(dst[row - 1, col - 1])) / (2 * h)
                    uy = (float(dst[row, col]) - float(dst[row, col - 1])) / h
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c4 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c4 = 1 / grad_u

                    dst[row, col] = (u0[row, col] + (1 / (_lambda * h * h)) * (
                                c1 * dst[row + 1, col] + c2 * dst[row - 1, col] + c3 * dst[
                            row, col + 1] + c4 * dst[row, col - 1])) * (
                                                         1 / (1 + (1 / (_lambda * h * h) * (c1 + c2 + c3 + c4))))
            # 处理边缘
            for row in range(1,image_width-1):
                dst[row,0] = dst[row,1]
                dst[row,image_height-1] = dst[row,image_height-1-1]
            for col in range(1,image_height-1):
                dst[0,col] = dst[1,col]
                dst[image_width-1,col] = dst[image_width-1-1,col]

            dst[0,0] = dst[1,1]
            dst[0,image_height-1] = dst[1,image_height-1-1]
            dst[image_width-1,0] = dst[image_width-1-1,1]
            dst[image_width-1,image_height-1] = dst[image_width-1-1,image_height-1-1]

            energy = 0
            for row in range(1, image_width - 1):
                for col in range(1, image_height - 1):
                    ux = (float(dst[row+1,col]) - float(dst[row,col]))/h
                    uy = (float(dst[row,col+1]) - float(dst[row,col]))/h
                    tmp = (float(u0[row,col]) - float(dst[row,col]))
                    fid = tmp*tmp
                    energy += math.sqrt(ux*ux + uy*uy) + _lambda*fid
            Energy.append(energy)
        end_time = time.time()
        print('Total Variation Minimization Complete. Time:{}'.format((end_time - start_time)))
        method_noise = src - dst
        
        self.results.append(dst)
        self.method_noise.append(method_noise)
        self.method_names.append("Total Variation Minimization")

        return dst,method_noise

    def Yaroslavsky_Filtering(self,src,dst=[],kernel_size=3,h=1):
        """
        @description  : Yaroslavsky filtering
        @param  : src : source image
        @param  : dst : destination/ result image
        @Returns  : Yaroslavsky filter result
        """
        print("Yaroslavsky Filtering start. Kernel size = {}, h = {}".format(kernel_size, h))
        start_time = time.time()
        image_width = src.shape[0]
        image_height = src.shape[1]
        weight = np.zeros(src.shape).astype('float64')
        dst = np.zeros(src.shape).astype('float64')
        padding_size = int((kernel_size - 1) / 2)
        padded_img = np.pad(src, padding_size, 'symmetric').astype('float64')
        for row in range(0, image_width):
            for col in range(0, image_height):
                sum = 0
                for i in range(-padding_size, padding_size + 1):
                    for j in range(-padding_size, padding_size + 1):
                        if i == 0 and j == 0:
                            continue
                        sum += np.exp(-(padded_img[(row + i), (col + j)] - padded_img[row,col])**2/(h*h))
                weight[row,col] = sum

        for row in range(padding_size, image_width - padding_size):
            for col in range(padding_size, image_height - padding_size):
                sum = 0
                sum_weight = 0
                for i in range(-padding_size, padding_size + 1):
                    for j in range(-padding_size, padding_size + 1):
                        sum += weight[(row+i),(col+j)]*int(src[(row+i),(col+j)])
                        sum_weight += weight[(row+i),(col+j)]
                dst[row,col] = sum/sum_weight
        end_time = time.time()
        print('Yaroslavsky Filtering Complete. Time:{}'.format(end_time - start_time))
        method_noise = src - dst

        self.results.append(dst)
        self.method_noise.append(method_noise)
        self.method_names.append("Yaroslavsky Filtering")
        return dst, method_noise

    def NLMeans(self, src, dst=[], t=10, f=3, h=1):
        """
        @description  : Non local means algorithm
        @param  : t : radius of search window
        @param  : f : radius of similarity window 
        @Returns  :
        """
        # t: radius of search window
        # f:radius of similarity window
        # H:degree of filtering
        print("Non-Local Means start. Radius of search window = {}, Radius of similarity window = {}, h = {}".format(t,f, h))
        start_time = time.time()
        width, height = np.shape(src)[0], np.shape(src)[1]
        dst = np.zeros((width,height),dtype='float64')
        padded_img = np.pad(src,((f,f),(f,f)) , 'edge').astype('float64')

        kernel = self.GaussianTemplate(2*f+1,1)
        pbar = tqdm(total=width*height)
        for x in range(0, width):
            for y in range(0, height):
                pbar.update(1)
                x1 = x + f
                y1 = y + f
                W1 = padded_img[x1 - f : x1 + f + 1, y1 - f : y1 + f + 1]
                # print(x1-f,x1+f)
                wmax = 0
                average = 0
                sweight = 0
                rmin = max(x1 - t, f + 1)
                rmax = min(x1 + t, width + f)
                smin = max(y1 - t, f + 1)
                smax = min(y1+ t, height + f)

                for r in range(rmin-1, rmax):
                    for s in range(smin-1, smax):
                        if (r == x1 and s == y1):
                            continue
                        W2 = padded_img[r - f: r+ f + 1, s-f : s + f + 1]
                        dis = np.sum(np.square(kernel*(W1-W2)))
                        w = np.exp(-dis / (h*h))
                        if w > wmax:
                            wmax = w
                        sweight = sweight + w
                        average = average + w * padded_img[r][s]
                average = average + wmax * padded_img[x1][y1]
                sweight = sweight + wmax
                if sweight > 0:
                    dst[x][y] = average / sweight
                else:
                    dst[x][y] = src[x][y]
        end_time = time.time()
        pbar.close()
        print('Non Local Means Complete. Time:{}'.format(end_time - start_time))
       

        method_noise  =  src - dst

        self.results.append(dst)
        self.method_noise.append(method_noise)
        self.method_names.append("Non-Local Means")
        return dst, method_noise
        
    
    def omp(D, data, sparsity):
        '''
        Given D，Y; Calculate X
        Input:
            D - dictionary (m, n-dims)
            data - Y: all patches (m, num-sample)
            sparsity - sparsity of x
        Output:
            X - sparse vec X: (n-dims, num-sample)
        '''
        X = np.zeros((D.shape[1], data.shape[1]))  # (n-dims, num-sample)
        tot_res = 0   # collect normed residual from every y-D@x
        # go through all data patches
        for i in range(data.shape[1]):
            # for all num-samples, every sample will have k-sparsity
            # every loop, finish one sample x
            ################### process bar ########################
            count = np.floor((i + 1) / float(data.shape[1]) * 100)
            sys.stdout.write("\r- omp Sparse coding : Channel : %d%%" % count)
            sys.stdout.flush()
            #######################################################
            #
            y = data[:, i]  # ith sample y, corresponding to ith x - (m,1)
            res = y  # initial residual
            omega = []
            res_norm = np.linalg.norm(res)
            xtemp_sparse = np.zeros(D.shape[1])  # (500,)

            while len(omega) < sparsity:
                # loop until x has sparsity-sparse (k-sparsity)
                # every loop, find one more sparse element
                proj = D.T @ res  # projection: find the max correlation between residual&D
                i_til = np.argmax(np.abs(proj))  # max correlation column
                omega.append(i_til)
                xtemp_sparse = np.linalg.pinv(D[:,omega])@y   # x = D^-1 @ y
                d_omg = D[:, omega]                  # (m, columns now have)
                recover_alr_y = d_omg @ xtemp_sparse  # y_til now can recover
                res = y - recover_alr_y           # calculate residual left
                res_norm = np.linalg.norm(res)  # update norm residual of this x

            tot_res += res_norm
            # update xi
            if len(omega) > 0:
                X[omega, i] = xtemp_sparse
        print('\r Sparse coding finished.\n')
        return X


    def initiate_D(patches, dict_size):
        '''
        dictionary intialization
        assign data columns to dictionary at random
        :param patches: (m, num of samples)
        :param dict_size: n-dims - then this would be the dimension of sparse vector x
        :return:
        D: normalized dictionary D
        '''
        # random select n-dims columns index
        indices = np.random.random_integers(0, patches.shape[1] - 1, dict_size)  # (500,)
        # choose the n-dims columns in Y as initial D
        D = np.array(patches[:, indices])  # select n-dims patches

        return D - D.mean()  # return normalized dictionary


    # update dictionary and sparse representations after sparse coding
    def update_D(D, data, X, j):
        '''
        Input:
            D - Dictionary (m, n-dims)
            data - Y all patches (m, num of samples)
            X: sparse matrix for x。(n-dims, num of samples) 每个patch变成了500维的稀疏向量，有8836个patch。
            j: now update the jth column of D
        Output:
            D_temp: new dictionary
            X: X would be updated followed by D
        '''
        indices = np.where(X[j, :] != 0)[0]  # find all x contributed to the i_til column
        D_temp = D  # work on new dictionary
        X_temp = X[:, indices]  # all x contributed to the i_til column

        if len(indices) > 1:
            # there're x contribute to this column
            X_temp[j, :] = 0  # set X's i_til row element to 0. remove the contribute to this column
            # ek: Y - D@X_temp: the contribution only of i_til column
            e_k = data[:, indices] - D_temp @ X_temp  # (m, a couple of columns)
            # make ek to be 2D matrix. (if only have 1 column, e_k would be a 1d array)
            u, s, vt = np.linalg.svd(np.atleast_2d(e_k))  # SVD error
            u = u[:,0]         # the first one
            s = s[0]            # largest one
            vt = vt[0,:]        # the first one
            D_temp[:, j] = u  # update dictionary with first column
            X[j, indices] = s * vt  # update x the sparse representations
        else:
            # no x have non-zero element corresponding to this column
            pass

        return D_temp, X
    def sliding_image(arr_in, patch_size, step=1):
        """
        Input
            arr_in : ndarray. N-d input array.
            patch_size : integer. sliding window size
            step : stride of sliding window
        Returns
            arr_out : All patches. (num, num, patch_size, patch_size)
        """
        # image size
        m_size, n_size = arr_in.shape
        # number of patches
        r = (m_size - patch_size) // step + 1
        c = (n_size - patch_size) // step + 1
        # all patches
        arr_out = np.zeros((r,c,patch_size,patch_size))
        for i in range(r):
            for j in range(c):
                rpos = i * step
                cpos = j * step
                # select patches
                arr_out[i,j] = arr_in[rpos:rpos+patch_size, cpos:cpos+patch_size]
        return arr_out    
        
    def getImagePatches(img, stride,patch_size = 7):
        '''
        Input：
            - img: image matrix
            - stride：stride of sliding window
        Return：
            - patches - (m,n) √m*√m-size's sliding window sampling patches.
                - m: window size vec
                - n: number of samples
            - patch - (r, c, √m, √m)
                - r: number of window patches on vertical direction
                - c: number of window patches on horizontal direction
                - √m: window size
                e.g. - (44,44, 7,7) 44*44 patches，each patch is 7*7 size
        '''
        # get indices of each patch from image matrix
        # This is also the R_{ij} marix mentioned in the paper
        patch_indices = NLM.sliding_image(img, patch_size, step=stride) # 返回 (r,c)个(window_shape)的数组(94,94,7,7)

        r,c= patch_indices.shape[0:2]  # window matrix size: r*c sliding patches
        i_r,i_c=patch_indices.shape[2:4]  # image patch size
        patches = np.zeros((i_r*i_c, r*c)) # every column is a patch
        # extract each image patchlena.png
        for i in range(r):
            for j in range(c):
                # extend patch to a vec -〉 (7*7, 44*44)
                patches[:, j+patch_indices.shape[1]*i] = np.concatenate(patch_indices[i, j], axis=0)
        return patches, patch_indices.shape
        
    def k_svd(patches, dict_size, sparsity,ksvd_iter):
        '''
        :param patches: patches from image (m, num of samples)
        :param dict_size: n-dims of every x
        :param sparsity: sparsity of every x
        :return:
            D: final dictionary
            X: corresponding X matrix (perhaps not sparse, so need omp to update again)
        '''
        # initial dictionary D
        D = NLM.initiate_D(patches, dict_size)
        # initializing sparse matrix: X
        X = np.zeros((D.T.dot(patches)).shape)  # (n-dims, num of samples)

        for k in range(ksvd_iter):  # ksvd_iter = 1
            print("KSVD Iter: {}/{} ".format(k + 1, ksvd_iter))
            # E step， update X
            X = NLM.omp(D, patches, sparsity)  # (n-dims, num of samples)
            # M step，update D
            count = 1
            dict_elem_order = np.random.permutation(D.shape[1])  # (0 ~ n-dims-1) array
            # get order of column elements
            for j in dict_elem_order:
                # update D column by column
                ################## process bar ###############################
                r = np.floor(count / float(D.shape[1]) * 100)
                sys.stdout.write("\r- k_svd Dictionary updating : %d%%" % r)
                sys.stdout.flush()
                ##############################################################
                # calculate the jth column
                D, X = NLM.update_D(D, patches, X, j)
                count += 1
            print("\nDictionary updating  finished")
        return D, X
    def reconstruct_image(patch_final, noisy_image,patch_size,sigma =20):
        '''
        :param patch_final: recovered patches i.g. (m, num)
        :param noisy_image: noisy image
        :return:
            img_out: denoised image
        '''
        # image temp
        img_out = np.zeros(noisy_image.shape)
        # weight temp
        weight = np.zeros(noisy_image.shape)
        num_blocks = noisy_image.shape[0] - patch_size + 1
        
        print('img_out',img_out.shape)
        print('patch_final',patch_final.shape)
        print('patch_size',patch_size)
        print('num_blocks',num_blocks)
        
        
        for l in range(patch_final.shape[1]):
            # put all patches back to a whole image
            i, j = divmod(l, num_blocks)
            # print(i, j)
            temp_patch = patch_final[:, l].reshape((patch_size,patch_size))
            img_out[i:(i+patch_size), j:(j+patch_size)] = img_out[i:(i+patch_size), j:(j+patch_size)] + temp_patch
            weight[i:(i+patch_size), j:(j+patch_size)] = weight[i:(i+patch_size), j:(j+patch_size)] + np.ones((patch_size,patch_size))

        # average all patches
        img_out = (noisy_image+0.034*sigma*img_out)/(1+0.034*sigma*weight)

        return img_out
        
        
    def K_SVD_main(self,img_noisy, dict_size, sparsity,window_stride,ksvd_iter,patch_size = 7):
        '''
        Input:
            img_noisy: input image
            dict_size: n-dims
            sparsity: sparsity of x
        Return:
            denoised_image: denoised image
        '''
        # generate noisy patches.
          
        stride = window_stride
        # get patches
        patches, _ = NLM.getImagePatches(img_noisy, stride)
        patches = patches - patches.mean()

        # K-SVD.
        dict_final, sparse_init = NLM.k_svd(patches, dict_size, sparsity,ksvd_iter)

        # omp
        ## preprocessing
        noisy_patches, noisy_patches_shape = NLM.getImagePatches(img_noisy, stride=1)
        data_mean = noisy_patches.mean()
        noisy_patches = noisy_patches - data_mean

        sparse_final = NLM.omp(dict_final, noisy_patches, sparsity)

        # Reconstruct the image.
        patches_approx = np.dot(dict_final, sparse_final) + data_mean
        
        denoised_image = NLM.reconstruct_image(patches_approx, img_noisy,patch_size)
        
        method_noise  =  img_noisy - denoised_image
        
        self.results.append(denoised_image)
        self.method_noise.append(method_noise)
        self.method_names.append("K-SVD")
        return denoised_image  ,method_noise