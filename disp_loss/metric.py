# from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from re import I
import torch
from torch import nn
import copy
import numpy as np



class One_PA(nn.Module):
    def __init__(self, device_type='cpu', dtype=torch.float32):
        super(One_PA, self).__init__()

        self.device_type = device_type
        self.dtype = dtype      # SSIM in half precision could be inaccurate

    def forward(self, input, target, maxdisp):
        """Implementation adopted from skimage.metrics.structural_similarity
        Default arguments set to multichannel=True, gaussian_weight=True, use_sample_covariance=False
        """

        input = input.squeeze(1).to(self.device_type)
        target = target.to(self.device_type)

        true_disp = copy.deepcopy(target)
        index = np.argwhere((true_disp > 0) & (true_disp < maxdisp))


        true_disp[index[0][:], index[1][:], index[2][:]] = np.abs(
            target[index[0][:], index[1][:], index[2][:]] - input[index[0][:], index[1][:], index[2][:]])
        
        correct_1 = (true_disp[index[0][:], index[1][:], index[2][:]] < 1)

        return (1 - (float(torch.sum(correct_1)) / float(len(index[0])))) * 100

class Two_PA(nn.Module):
    def __init__(self, device_type='cpu', dtype=torch.float32):
        super(Two_PA, self).__init__()

        self.device_type = device_type
        self.dtype = dtype      # SSIM in half precision could be inaccurate

    def forward(self, input, target, maxdisp):
        """Implementation adopted from skimage.metrics.structural_similarity
        Default arguments set to multichannel=True, gaussian_weight=True, use_sample_covariance=False
        """

        input = input.squeeze(1).to(self.device_type)
        target = target.to(self.device_type)

        true_disp = copy.deepcopy(target)
        index = np.argwhere((true_disp > 0) & (true_disp < maxdisp))


        true_disp[index[0][:], index[1][:], index[2][:]] = np.abs(
            target[index[0][:], index[1][:], index[2][:]] - input[index[0][:], index[1][:], index[2][:]])
        
        correct_1 = (true_disp[index[0][:], index[1][:], index[2][:]] < 2) 

        return (1 - (float(torch.sum(correct_1)) / float(len(index[0])))) * 100

class Three_PA(nn.Module):
    def __init__(self, device_type='cpu', dtype=torch.float32):
        super(Three_PA, self).__init__()

        self.device_type = device_type
        self.dtype = dtype      # SSIM in half precision could be inaccurate

    def forward(self, input, target, maxdisp):
        """Implementation adopted from skimage.metrics.structural_similarity
        Default arguments set to multichannel=True, gaussian_weight=True, use_sample_covariance=False
        """

        input = input.squeeze(1).to(self.device_type)
        target = target.to(self.device_type)

        true_disp = copy.deepcopy(target)
        index = np.argwhere((true_disp > 0) & (true_disp < maxdisp))


        true_disp[index[0][:], index[1][:], index[2][:]] = np.abs(
            target[index[0][:], index[1][:], index[2][:]] - input[index[0][:], index[1][:], index[2][:]])
        
        correct_1 = (true_disp[index[0][:], index[1][:], index[2][:]] < 3) 
        return (1 - (float(torch.sum(correct_1)) / float(len(index[0])))) * 100

class MAE(nn.Module):
    def __init__(self, device_type='cpu', dtype=torch.float32):
        super(MAE, self).__init__()

        self.device_type = device_type
        self.dtype = dtype      # SSIM in half precision could be inaccurate

    def forward(self, input, target, maxdisp):
        """Implementation adopted from skimage.metrics.structural_similarity
        Default arguments set to multichannel=True, gaussian_weight=True, use_sample_covariance=False
        """

        input = input.squeeze(1).to(self.device_type)
        target = target.to(self.device_type)

        true_disp = copy.deepcopy(target)
        index = np.argwhere((true_disp > 0) & (true_disp < maxdisp))

        true_disp[index[0][:], index[1][:], index[2][:]] = np.abs(
            target[index[0][:], index[1][:], index[2][:]] - input[index[0][:], index[1][:], index[2][:]])
        
        mae = torch.mean(true_disp[index[0][:], index[1][:], index[2][:]])
        
        return mae

class RMSE(nn.Module):
    def __init__(self, device_type='cpu', dtype=torch.float32):
        super(RMSE, self).__init__()

        self.device_type = device_type
        self.dtype = dtype      # SSIM in half precision could be inaccurate

    def forward(self, input, target, maxdisp):
        """Implementation adopted from skimage.metrics.structural_similarity
        Default arguments set to multichannel=True, gaussian_weight=True, use_sample_covariance=False
        """

        input = input.squeeze(1).to(self.device_type)
        target = target.to(self.device_type)

        true_disp = copy.deepcopy(target)
        index = np.argwhere((true_disp > 0) & (true_disp < maxdisp))

        true_disp[index[0][:], index[1][:], index[2][:]] = np.abs(
            target[index[0][:], index[1][:], index[2][:]] - input[index[0][:], index[1][:], index[2][:]])
 
        # rmse =  torch.sqrt(torch.sum(torch.mul(true_disp[index[0][:], index[1][:], index[2][:]], \
        #      true_disp[index[0][:], index[1][:], index[2][:]]))/true_disp[index[0][:], index[1][:], index[2][:]].size()[0])
        rmse =  torch.sqrt(torch.sum(torch.mul(true_disp[index[0][:], index[1][:], index[2][:]], \
             true_disp[index[0][:], index[1][:], index[2][:]])) / float(len(index[0])))
        return rmse

def disparity_to_depth(disparity_image):
    
    # unknown_disparity = disparity_image == float('inf')
    unknown_disparity = disparity_image == 0.0
    depth_image = \
        0.6 / (disparity_image + 1e-7)
    depth_image[unknown_disparity] = float('inf')
    return depth_image

def compute_absolute_error(estimated_disparity,
                           ground_truth_disparity,
                           use_mean=True):

    absolute_difference = (estimated_disparity - ground_truth_disparity).abs()
    locations_without_ground_truth = torch.isinf(ground_truth_disparity)
    pixelwise_absolute_error = absolute_difference.clone()
    pixelwise_absolute_error[locations_without_ground_truth] = 0
    absolute_differece_with_ground_truth = absolute_difference[
        ~locations_without_ground_truth]
    if absolute_differece_with_ground_truth.numel() == 0:
        average_absolute_error = 0.0
    else:
        if use_mean:
            average_absolute_error = absolute_differece_with_ground_truth.mean(
            ).item()
        else:
            average_absolute_error = absolute_differece_with_ground_truth.median(
            ).item()

    return pixelwise_absolute_error, average_absolute_error

class Mean_Depth(nn.Module):
    def __init__(self, device_type = 'cpu', dtype = torch.float32):
        super(Mean_Depth, self).__init__()
        self.device_type = device_type
        self.dtype = dtype 
    
    def forward(self, input, target, maxdisp):
        input = input.squeeze(1).to(self.device_type)
        target = target.to(self.device_type)

        input = disparity_to_depth(input)
        target = disparity_to_depth(target)

        error = compute_absolute_error(input, target)[1] * 100.0

        return error

class Mean_Disp(nn.Module):
    def __init__(self, device_type = 'cpu', dtype = torch.float32):
        super(Mean_Disp, self).__init__()
        self.device_type = device_type
        self.dtype = dtype 
    
    def forward(self, input, target, maxdisp):
        input = input.squeeze(1).to(self.device_type)
        target = target.to(self.device_type)

        unknown_disparity = target == 0.0
        target[unknown_disparity] = float('inf')
        error = compute_absolute_error(input, target)[1]

        return error
