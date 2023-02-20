# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule



class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)



@HEADS.register_module()
class DepthwiseSeparableASPPHeadCAC(ASPPHead):

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(DepthwiseSeparableASPPHeadCAC, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.proj = nn.Sequential(
            nn.Linear(self.channels*2, self.channels//2, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(self.channels//2, self.channels),
        )                 
        self.apd_proj = nn.Sequential(
            nn.Linear(self.channels*2, self.channels//2, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(self.channels//2, self.channels),
        )                 
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)      
        self.dropout = nn.Dropout2d(self.dropout_ratio)         
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)


    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        feat = self.sep_bottleneck(output)
        # output = self.cls_seg(output)
        # return output
        out = self.conv_seg(self.dropout(feat))
        return out, feat

    def get_adaptive_perspective(self, x, y, new_proto, proto):
        raw_x = x.clone()
        # y: [b, h, w]
        # x: [b, c, h, w]
        b, c, h, w = x.shape[:]
        y = F.interpolate(y.float(), size=(h, w), mode='nearest')  # b, 1, h, w
        unique_y = list(y.unique())
        if 255 in unique_y:
            unique_y.remove(255)
        # new_proto = self.conv_seg[1].weight.detach().data.squeeze() # [cls, 512]
        tobe_align = []
        label_list = []
        for tmp_y in unique_y:
            tmp_mask = (y == tmp_y).float()
            tmp_proto = (x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)
            onehot_vec = torch.zeros(new_proto.shape[0], 1).cuda()  # cls, 1
            onehot_vec[tmp_y.long()] = 1
            new_proto = new_proto * (1 - onehot_vec) + tmp_proto.unsqueeze(0) * onehot_vec
            tobe_align.append(tmp_proto.unsqueeze(0))
            label_list.append(tmp_y)  

        # new_proto: [cls, 512]
        
        new_proto = torch.cat([new_proto, proto], -1)
        new_proto = self.apd_proj(new_proto)
        new_proto = new_proto.unsqueeze(-1).unsqueeze(-1)   # cls, 512, 1, 1
        new_proto = F.normalize(new_proto, 2, 1)
        raw_x = F.normalize(raw_x, 2, 1)
        pred = F.conv2d(raw_x, weight=new_proto) * 15
        return pred

    def get_pred(self, x, proto):
        b, c, h, w = x.size()[:]
        if len(proto.shape[:]) == 3:
            # x: [b, c, h, w]
            # proto: [b, cls, c]  
            cls_num = proto.size(1)
            x = x / torch.norm(x, 2, 1, True)
            proto = proto / torch.norm(proto, 2, -1, True)  # b, n, c
            x = x.contiguous().view(b, c, h*w)  # b, c, hw
            pred = proto @ x  # b, cls, hw
        elif len(proto.shape[:]) == 2:
            # x: [b, c, h, w]
            # proto: [cls, c]            
            cls_num = proto.size(0)
            x = x / torch.norm(x, 2, 1, True)
            proto = proto / torch.norm(proto, 2, 1, True)
            x = x.contiguous().view(b, c, h*w)  # b, c, hw
            proto = proto.unsqueeze(0)  # 1, cls, c
            pred = proto @ x  # b, cls, hw
        pred = pred.contiguous().view(b, cls_num, h, w)
        return pred * 15

    def post_refine_proto_v2(self, x, pred, proto):
        # pred: [b, n, h, w]
        # raw_x: [b, c, h, w]
        # proto: [n, c]
        # pred_proto: [n, c]
        raw_x = x.clone()        
        b, c, h, w = raw_x.shape[:]
        pred = pred.view(b, proto.shape[0], h*w)
        pred = F.softmax(pred, 1)   # b, n, hw
        pred_proto = (pred @ raw_x.view(b, c, h*w).permute(0, 2, 1)) / (pred.sum(-1).unsqueeze(-1) + 1e-12)

        pred_proto = torch.cat([pred_proto, proto.unsqueeze(0).repeat(pred_proto.shape[0], 1, 1)], -1)  # b, n, 2c
        pred_proto = self.proj(pred_proto)
        new_pred = self.get_pred(raw_x, pred_proto)
        return new_pred

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        y = gt_semantic_seg
        x, feat = self.forward(inputs)

        pre_self_x = x.clone()
        x = self.post_refine_proto_v2(x=feat, pred=x, proto=self.conv_seg.weight.squeeze())      
        apd_pred = self.get_adaptive_perspective(x=feat, y=y, new_proto=self.conv_seg.weight.detach().data.squeeze(), proto=self.conv_seg.weight.squeeze())   

        kl_loss = get_distill_loss(pred=x, soft=apd_pred.detach(), target=y.squeeze(1))

        pre_self_x = F.interpolate(pre_self_x, size=y.shape[-2:], mode='bilinear', align_corners=True)
        pre_self_loss = self.criterion(pre_self_x, y.squeeze(1).long()) 
        apd_pred = F.interpolate(apd_pred, size=y.shape[-2:], mode='bilinear', align_corners=True)
        pre_loss = self.criterion(apd_pred, y.squeeze(1).long()) 

        losses = self.losses(x, y)
        losses['PreSelfLoss'] =  pre_self_loss.detach().data
        losses['PreLoss'] =  pre_loss.detach().data
        losses['KLLoss'] =  kl_loss.detach().data
        losses['MainLoss'] =  losses['loss_ce'].detach().data
        losses['loss_ce'] = losses['loss_ce'] + pre_self_loss + pre_loss + kl_loss
        return losses 


    def forward_test(self, inputs, img_metas, test_cfg):
        x, feat = self.forward(inputs)
        x = self.post_refine_proto_v2(x=feat, pred=x, proto=self.conv_seg.weight.squeeze())      
        return x     






def get_distill_loss(pred, soft, target, smoothness=0.5, eps=0):
    '''
    knowledge distillation loss
    '''
    b, c, h, w = soft.shape[:]
    soft.detach()
    target = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[-2:], mode='nearest').squeeze(1).long()
    onehot = target.view(-1, 1) # bhw, 1
    ignore_mask = (onehot == 255).float()
    onehot = onehot * (1 - ignore_mask) 
    onehot = torch.zeros(b*h*w, c).cuda().scatter_(1,onehot.long(),1)  # bhw, n
    onehot = onehot.contiguous().view(b, h, w, c).permute(0, 3, 1, 2)   # b, n, h, w
    sm_soft = F.softmax(soft / 1, 1)
    smoothed_label = smoothness * sm_soft + (1 - smoothness) * onehot
    if eps > 0: 
        smoothed_label = smoothed_label * (1 - eps) + (1 - smoothed_label) * eps / (smoothed_label.shape[1] - 1) 

    loss = torch.mul(-1 * F.log_softmax(pred, dim=1), smoothed_label)   # b, n, h, w
    
    sm_soft = F.softmax(soft / 1, 1)   # b, c, h, w    
    entropy_mask = -1 * (sm_soft * torch.log(sm_soft + 1e-12)).sum(1)
    loss = loss.sum(1) 

    ### for class-wise entropy estimation    
    unique_classes = list(target.unique())
    if 255 in unique_classes:
        unique_classes.remove(255)
    valid_mask = (target != 255).float()
    entropy_mask = entropy_mask * valid_mask
    loss_list = []
    weight_list = []
    for tmp_y in unique_classes:
        tmp_mask = (target == tmp_y).float()
        tmp_entropy_mask = entropy_mask * tmp_mask
        class_weight = 1
        tmp_loss = (loss * tmp_entropy_mask).sum() / (tmp_entropy_mask.sum() + 1e-12)
        loss_list.append(class_weight * tmp_loss)
        weight_list.append(class_weight)
    if len(weight_list) > 0:
        loss = sum(loss_list) / (sum(weight_list) + 1e-12)
    else:
        loss = torch.zeros(1).cuda().mean()
    return loss