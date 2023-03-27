from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import math
import numpy as np
import torchvision.transforms as transforms
import PIL
import os
import matplotlib.pyplot as plt


class AffinityFeature(nn.Module):
    def __init__(self, win_h, win_w, dilation, cut):
        super(AffinityFeature, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.dilation = dilation
        self.cut = 0

    def padding(self, x, win_h, win_w, dilation):
        pad_t = (win_w // 2 * dilation, win_w // 2 * dilation,
                 win_h // 2 * dilation, win_h // 2 * dilation)
        out = F.pad(x, pad_t, mode='constant')
        return out

    def forward(self, feature):
        B, C, H, W = feature.size()
        feature = F.normalize(feature, dim=1, p=2)

        # affinity = []
        # pad_feature = self.padding(feature, win_w=self.win_w, win_h=self.win_h, dilation=self.dilation)
        # for i in range(self.win_w):
        #     for j in range(self.win_h):
        #         if (i == self.win_w // 2) & (j == self.win_h // 2):
        #             continue
        #         simi = self.cal_similarity(
        #             pad_feature[:, :, self.dilation*j:self.dilation*j+H, self.dilation*i:self.dilation*i+W],
        #             feature, self.simi_type)
        #
        #         affinity.append(simi)
        # affinity = torch.stack(affinity, dim=1)
        #
        # affinity[affinity < self.cut] = self.cut

        unfold_feature = nn.Unfold(
            kernel_size=(self.win_h, self.win_w), dilation=self.dilation, padding=self.dilation)(feature)
        all_neighbor = unfold_feature.reshape(B, C, -1, H, W).transpose(1, 2)
        num = (self.win_h * self.win_w) // 2
        
        neighbor = torch.cat((all_neighbor[:, :num], all_neighbor[:, num+1:]), dim=1)
        feature = feature.unsqueeze(1)
        affinity = torch.sum(neighbor * feature, dim=2)

        affinity[affinity < self.cut] = self.cut
        # import pdb; pdb.set_trace()
        return affinity

class AffinityImageEvent(nn.Module):
    def __init__(self, win_h, win_w, dilation, cut):
        super(AffinityImageEvent, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.dilation = dilation
        self.cut = 0

    def padding(self, x, win_h, win_w, dilation):
        pad_t = (win_w // 2 * dilation, win_w // 2 * dilation,
                 win_h // 2 * dilation, win_h // 2 * dilation)
        out = F.pad(x, pad_t, mode='constant')
        return out

    def forward(self, image, event):
        B, C, H, W = image.size()
        image = F.normalize(image, dim=1, p=2)
        event = F.normalize(event, dim=1, p=2)

        # affinity = []
        # pad_feature = self.padding(feature, win_w=self.win_w, win_h=self.win_h, dilation=self.dilation)
        # for i in range(self.win_w):
        #     for j in range(self.win_h):
        #         if (i == self.win_w // 2) & (j == self.win_h // 2):
        #             continue
        #         simi = self.cal_similarity(
        #             pad_feature[:, :, self.dilation*j:self.dilation*j+H, self.dilation*i:self.dilation*i+W],
        #             feature, self.simi_type)
        #
        #         affinity.append(simi)
        # affinity = torch.stack(affinity, dim=1)
        #
        # affinity[affinity < self.cut] = self.cut

        unfold_image = nn.Unfold(
            kernel_size=(self.win_h, self.win_w), dilation=self.dilation, padding=self.dilation)(image)
        image_neighbor = unfold_image.reshape(B, C, -1, H, W).transpose(1, 2)
        unfold_event = nn.Unfold(
            kernel_size=(self.win_h, self.win_w), dilation=self.dilation, padding=self.dilation)(event)
        event_neighbor = unfold_event.reshape(B, C, -1, H, W).transpose(1, 2)
        # num = (self.win_h * self.win_w) // 2
        # neighbor = torch.cat((image_neighbor[:, :num], image_neighbor[:, num+1:]), dim=1)
        
        

        # image = image.unsqueeze(1)
        affinity = torch.sum(image_neighbor * event_neighbor, dim=2)

        affinity[affinity < self.cut] = self.cut
        # import pdb; pdb.set_trace()
        return affinity

class AffinityImageEvent2(nn.Module):
    def __init__(self, win_h, win_w, dilation, cut):
        super(AffinityImageEvent2, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.dilation = dilation
        self.cut = 0

    def padding(self, x, win_h, win_w, dilation):
        pad_t = (win_w // 2 * dilation, win_w // 2 * dilation,
                 win_h // 2 * dilation, win_h // 2 * dilation)
        out = F.pad(x, pad_t, mode='constant')
        return out

    def forward(self, image, event):
        B, C, H, W = image.size()
        image = F.normalize(image, dim=1, p=2)
        event = F.normalize(event, dim=1, p=2)

        # affinity = []
        # pad_feature = self.padding(feature, win_w=self.win_w, win_h=self.win_h, dilation=self.dilation)
        # for i in range(self.win_w):
        #     for j in range(self.win_h):
        #         if (i == self.win_w // 2) & (j == self.win_h // 2):
        #             continue
        #         simi = self.cal_similarity(
        #             pad_feature[:, :, self.dilation*j:self.dilation*j+H, self.dilation*i:self.dilation*i+W],
        #             feature, self.simi_type)
        #
        #         affinity.append(simi)
        # affinity = torch.stack(affinity, dim=1)
        #
        # affinity[affinity < self.cut] = self.cut

        unfold_image = nn.Unfold(
            kernel_size=(self.win_h, self.win_w), dilation=self.dilation, padding=self.dilation)(image)
        image_neighbor = unfold_image.reshape(B, C, -1, H, W).transpose(1, 2)
        unfold_event = nn.Unfold(
            kernel_size=(self.win_h, self.win_w), dilation=self.dilation, padding=self.dilation)(image)
        event_neighbor = unfold_event.reshape(B, C, -1, H, W).transpose(1, 2)
        # num = (self.win_h * self.win_w) // 2
        # neighbor = torch.cat((image_neighbor[:, :num], image_neighbor[:, num+1:]), dim=1)
        
        
        num = (self.win_h * self.win_w) // 2
        
        image_neighbor = torch.cat((image_neighbor[:, :num], image_neighbor[:, num+1:]), dim=1)
        image = image.unsqueeze(1)
        image_affinity = image_neighbor * image

        event_neighbor = torch.cat((event_neighbor[:, :num], event_neighbor[:, num+1:]), dim=1)
        event = event.unsqueeze(1)
        event_affinity = event_neighbor * event

        affinity = torch.sum(image_affinity * event_affinity, dim=2)

        affinity[affinity < self.cut] = self.cut
        # import pdb; pdb.set_trace()
        return affinity




def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

# class matchshifted(nn.Module):
#     def __init__(self):
#         super(matchshifted, self).__init__()
#
#     def forward(self, left, right, shift):
#         batch, filters, height, width = left.size()
#         shifted_left  = F.pad(torch.index_select(left,  3, Variable(torch.LongTensor([i for i in range(shift,width)])).cuda()),(shift,0,0,0))
#         shifted_right = F.pad(torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width-shift)])).cuda()),(shift,0,0,0))
#         out = torch.cat((shifted_left,shifted_right),1).view(batch,filters*2,1,height,width)
#         return out


class DisparityRegression(nn.Module):

    def __init__(self, maxdisp, win_size):
        super(DisparityRegression, self).__init__()
        self.max_disp = maxdisp
        self.win_size = win_size

    def forward(self, x):
        disp = torch.arange(0, self.max_disp).view(1, -1, 1, 1).float().to(x.device)

        if self.win_size > 0:
            max_d = torch.argmax(x, dim=1, keepdim=True)
            d_value = []
            prob_value = []
            for d in range(-self.win_size, self.win_size + 1):
                index = max_d + d
                index[index < 0] = 0
                index[index > x.shape[1] - 1] = x.shape[1] - 1
                d_value.append(index)

                prob = torch.gather(x, dim=1, index=index)
                prob_value.append(prob)

            part_x = torch.cat(prob_value, dim=1)
            part_x = part_x / (torch.sum(part_x, dim=1, keepdim=True) + 1e-8)
            part_d = torch.cat(d_value, dim=1).float()
            out = torch.sum(part_x * part_d, dim=1)

        else:
            out = torch.sum(x * disp, 1)

        return out


class feature_extraction(nn.Module):
    def __init__(self, affinity_settings, in_channel, structure_fc=4, fuse_mode='separate'):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.sfc = structure_fc
        self.fuse_mode = fuse_mode
        self.win_w = affinity_settings['win_w']
        self.win_h = affinity_settings['win_h']
        self.dilation = affinity_settings['dilation']

        self.firstconv = nn.Sequential(convbn(in_channel, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

        if self.sfc > 0:
            if fuse_mode == 'aggregate':
                self.embedding = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(128, 128, kernel_size=1, padding=0, stride=1, bias=False))

                in_c = self.win_w * self.win_h - 1
                self.sfc_conv1 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                               nn.ReLU(inplace=True))
                self.sfc_conv2 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                               nn.ReLU(inplace=True))
                self.sfc_conv3 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                               nn.ReLU(inplace=True))
                self.sfc_conv4 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                               nn.ReLU(inplace=True))

                self.lastconv = nn.Sequential(convbn(4*self.sfc, 32, 3, 1, 1, 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False))

                # self.lastconv = nn.Sequential(convbn(320 + 4*self.sfc, 128, 3, 1, 1, 1),
                #                               nn.ReLU(inplace=True),
                #                               nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False))

                self.to_sf = StructureFeature(affinity_settings, self.sfc)

            elif fuse_mode == 'separate':
                # self.embedding_l1 = nn.Sequential(convbn(32, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                #                                   nn.ReLU(inplace=True),
                #                                   nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
                # self.to_sf_l1 = StructureFeature(affinity_settings, self.sfc)

                self.embedding_l2 = nn.Sequential(convbn(64, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
                self.to_sf_l2 = StructureFeature(affinity_settings, self.sfc)

                self.embedding_l3 = nn.Sequential(convbn(128, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
                self.to_sf_l3 = StructureFeature(affinity_settings, self.sfc)

                self.embedding_l4 = nn.Sequential(convbn(128, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
                self.to_sf_l4 = StructureFeature(affinity_settings, self.sfc)

                # self.lastconv = nn.Sequential(convbn(3 * 4 * self.sfc, 32, 3, 1, 1, 1),
                #                               nn.ReLU(inplace=True),
                #                               nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1, bias=False))

                self.lastconv = nn.Sequential(convbn(320 + 3 * 4 * self.sfc, 128, 3, 1, 1, 1),
                                              nn.ReLU(inplace=True),
                                              
                                              nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

                # self.lastconv = nn.Sequential(convbn(320 + 3 * 4 * self.sfc, 320, 3, 1, 1, 1),
                #                               nn.ReLU(inplace=True),
                #                               nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output_l1 = self.layer1(output)
        output_l2 = self.layer2(output_l1)
        output_l3 = self.layer3(output_l2)
        output_l4 = self.layer4(output_l3)
        
        # output_l1 = F.interpolate(output_l1, (output_l4.size()[2], output_l4.size()[3]),
        #                           mode='bilinear', align_corners=True)

        cat_feature = torch.cat((output_l2, output_l3, output_l4), 1)

        if self.sfc > 0:
            if self.fuse_mode == 'aggregate':
                embedding = self.embedding(cat_feature)

                cat_sf, affinity = self.to_sf(embedding)

                # output_feature = self.lastconv(torch.cat((cat_feature, cat_sf), dim=1))
                output_feature = self.lastconv(cat_sf)

            elif self.fuse_mode == 'separate':
                # embedding_l1 = self.embedding_l1(output_l1)
                # l1_sf, l1_affi = self.to_sf_l1(embedding_l1)

                embedding_l2 = self.embedding_l2(output_l2.detach())
                l2_sf, l2_affi = self.to_sf_l2(embedding_l2)

                embedding_l3 = self.embedding_l3(output_l3.detach())
                l3_sf, l3_affi = self.to_sf_l3(embedding_l3)

                embedding_l4 = self.embedding_l4(output_l4.detach())
                l4_sf, l4_affi = self.to_sf_l4(embedding_l4)

                # output_feature = self.lastconv(torch.cat((l2_sf, l3_sf, l4_sf), dim=1))

                
                output_feature = self.lastconv(torch.cat((cat_feature, l2_sf, l3_sf, l4_sf), dim=1))
                affinity = torch.cat((l2_affi, l3_affi, l4_affi), dim=1)

            return output_feature, affinity

        else:
            output_feature = self.lastconv(cat_feature)

            return output_feature


class event_image_feature_extraction(nn.Module):
    def __init__(self, affinity_settings, structure_fc=4, fuse_mode='separate'):
        super(event_image_feature_extraction, self).__init__()
        self.inplanes = 32
        self.event_inplanes = 32
        self.sfc = structure_fc
        self.fuse_mode = fuse_mode
        self.win_w = affinity_settings['win_w']
        self.win_h = affinity_settings['win_h']
        self.dilation = affinity_settings['dilation']

        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.event_firstconv = nn.Sequential(convbn(32, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.event_layer1 = self._event_make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.event_layer2 = self._event_make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.event_layer3 = self._event_make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.event_layer4 = self._event_make_layer(BasicBlock, 128, 3, 1, 1, 2)



        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))


        self.embedding_l2 = nn.Sequential(convbn(64, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
        self.to_sf_l2 = EI_Structure_Feature(affinity_settings, self.sfc, 64)

        self.embedding_l3 = nn.Sequential(convbn(128, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
        self.to_sf_l3 = EI_Structure_Feature(affinity_settings, self.sfc, 128)

        self.embedding_l4 = nn.Sequential(convbn(128, 64, kernel_size=3, stride=1, pad=1, dilation=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
        self.to_sf_l4 = EI_Structure_Feature(affinity_settings, self.sfc, 128)

        self.lastconv = nn.Sequential(convbn(320 + 320, 480, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        
                                        nn.Conv2d(480, 320, kernel_size=1, padding=0, stride=1, bias=False))


    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _event_make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.event_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.event_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.event_inplanes, planes, stride, downsample, pad, dilation))
        self.event_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.event_inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, image, event):
        output = self.firstconv(image)
        output_l1 = self.layer1(output)
        output_l2 = self.layer2(output_l1)
        

        event_output = self.event_firstconv(event)
        event_l1 = self.event_layer1(event_output)
        event_l2 = self.event_layer2(event_l1)

        output_l2, event_l2 = self.to_sf_l2(output_l2, event_l2)

        output_l3 = self.layer3(output_l2)
        event_l3 = self.event_layer3(event_l2)

        output_l3, event_l3 = self.to_sf_l3(output_l3, event_l3)
        
        output_l4 = self.layer4(output_l3)
        event_l4 = self.event_layer4(event_l3)

        output_l4, event_l4 = self.to_sf_l4(output_l4, event_l4)

        cat_feature = torch.cat((output_l2, output_l3, output_l4, event_l2, event_l3, event_l4), 1)
        
        output_feature = self.lastconv(cat_feature)
        
        return output_feature

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class StructureFeature(nn.Module):
    def __init__(self, affinity_settings, sfc):
        super(StructureFeature, self).__init__()

        self.win_w = affinity_settings['win_w']
        self.win_h = affinity_settings['win_h']
        self.dilation = affinity_settings['dilation']

        self.sfc = sfc

        in_c = self.win_w * self.win_h - 1

        self.sfc_conv1 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv2 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv3 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv4 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))

    def forward(self, x):

        affinity1 = AffinityFeature(self.win_h, self.win_w, self.dilation[0], 0)(x)
        affinity2 = AffinityFeature(self.win_h, self.win_w, self.dilation[1], 0)(x)
        affinity3 = AffinityFeature(self.win_h, self.win_w, self.dilation[2], 0)(x)
        affinity4 = AffinityFeature(self.win_h, self.win_w, self.dilation[3], 0)(x)

        affi_feature1 = self.sfc_conv1(affinity1)
        affi_feature2 = self.sfc_conv2(affinity2)
        affi_feature3 = self.sfc_conv3(affinity3)
        affi_feature4 = self.sfc_conv4(affinity4)

        out_feature = torch.cat((affi_feature1, affi_feature2, affi_feature3, affi_feature4), dim=1)
        affinity = torch.cat((affinity1, affinity2, affinity3, affinity4), dim=1)

        # out_feature = torch.cat((affi_feature1, affi_feature2, affi_feature3), dim=1)
        # affinity = torch.cat((affinity1, affinity2, affinity3), dim=1)

        return out_feature, affinity


class EI_Structure_Feature(nn.Module):
    def __init__(self, affinity_settings, sfc, fea_ch):
        super(EI_Structure_Feature, self).__init__()

        self.win_w = affinity_settings['win_w']
        self.win_h = affinity_settings['win_h']
        self.dilation = affinity_settings['dilation']

        self.sfc = sfc
        self.fea_ch = fea_ch

        # in_c = self.win_w * self.win_h - 1
        in_c = self.win_w * self.win_h

        self.sfc_conv1 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv2 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv3 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv4 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))

        self.image_conv = nn.Sequential(convbn(self.fea_ch + self.sfc * 4, self.fea_ch, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.event_conv = nn.Sequential(convbn(self.fea_ch + self.sfc * 4, self.fea_ch, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))

    def forward(self, image, event):

        affinity1 = AffinityImageEvent(self.win_h, self.win_w, self.dilation[0], 0)(image, event)
        affinity2 = AffinityImageEvent(self.win_h, self.win_w, self.dilation[1], 0)(image, event)
        affinity3 = AffinityImageEvent(self.win_h, self.win_w, self.dilation[2], 0)(image, event)
        affinity4 = AffinityImageEvent(self.win_h, self.win_w, self.dilation[3], 0)(image, event)

        affi_feature1 = self.sfc_conv1(affinity1)
        affi_feature2 = self.sfc_conv2(affinity2)
        affi_feature3 = self.sfc_conv3(affinity3)
        affi_feature4 = self.sfc_conv4(affinity4)

        out_image_feature = self.image_conv(torch.cat((image, affi_feature1, affi_feature2, affi_feature3, affi_feature4), dim=1))
        out_event_feature = self.event_conv(torch.cat((event, affi_feature1, affi_feature2, affi_feature3, affi_feature4), dim=1))
        

        affinity = torch.cat((affinity1, affinity2, affinity3, affinity4), dim=1)

        # out_feature = torch.cat((affi_feature1, affi_feature2, affi_feature3), dim=1)
        # affinity = torch.cat((affinity1, affinity2, affinity3), dim=1)

        return out_image_feature, out_event_feature

class EI_Structure_Feature2(nn.Module):
    def __init__(self, affinity_settings, sfc, fea_ch):
        super(EI_Structure_Feature2, self).__init__()

        self.win_w = affinity_settings['win_w']
        self.win_h = affinity_settings['win_h']
        self.dilation = affinity_settings['dilation']

        self.sfc = sfc
        self.fea_ch = fea_ch

        in_c = self.win_w * self.win_h - 1
        # in_c = self.win_w * self.win_h

        self.sfc_conv1 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv2 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv3 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.sfc_conv4 = nn.Sequential(convbn(in_c, self.sfc, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))

        self.image_conv = nn.Sequential(convbn(self.fea_ch + self.sfc * 4, self.fea_ch, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))
        self.event_conv = nn.Sequential(convbn(self.fea_ch + self.sfc * 4, self.fea_ch, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True))

    def forward(self, image, event):

        affinity1 = AffinityImageEvent2(self.win_h, self.win_w, self.dilation[0], 0)(image, event)
        affinity2 = AffinityImageEvent2(self.win_h, self.win_w, self.dilation[1], 0)(image, event)
        affinity3 = AffinityImageEvent2(self.win_h, self.win_w, self.dilation[2], 0)(image, event)
        affinity4 = AffinityImageEvent2(self.win_h, self.win_w, self.dilation[3], 0)(image, event)

        affi_feature1 = self.sfc_conv1(affinity1)
        affi_feature2 = self.sfc_conv2(affinity2)
        affi_feature3 = self.sfc_conv3(affinity3)
        affi_feature4 = self.sfc_conv4(affinity4)

        out_image_feature = self.image_conv(torch.cat((image, affi_feature1, affi_feature2, affi_feature3, affi_feature4), dim=1))
        out_event_feature = self.event_conv(torch.cat((event, affi_feature1, affi_feature2, affi_feature3, affi_feature4), dim=1))
        

        affinity = torch.cat((affinity1, affinity2, affinity3, affinity4), dim=1)

        # out_feature = torch.cat((affi_feature1, affi_feature2, affi_feature3), dim=1)
        # affinity = torch.cat((affinity1, affinity2, affinity3), dim=1)

        return out_image_feature, out_event_feature

