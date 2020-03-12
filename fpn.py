import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
from model.utils.config import cfg    # rm 'lib.', or cfg will be create a new copy
from model.rpn.rpn_fpn import _RPN_FPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _smooth_l1_loss_epi, _smooth_l1_loss_penalty, _crop_pool_layer, _affine_grid_gen, _affine_theta
import time
import pdb

class _FPN(nn.Module):
    """ FPN """
    def __init__(self, classes, class_agnostic):
        super(_FPN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        #self.dropout = nn.Dropout(0.5)

        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # define rpn
        self.RCNN_rpn = _RPN_FPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # NOTE: the original paper used pool_size = 7 for cls branch, and 14 for mask branch, to save the
        # computation time, we first use 14 as the pool_size, and then do stride=2 pooling for cls branch.
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # custom weights initialization called on netG and netD
        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        normal_init(self.RCNN_toplayer, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer3, 0, 0.01, cfg.TRAIN.TRUNCATED)
#        normal_init(self.Loc_Uncertain, 0, 0.0001, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.1, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        img_area = im_info[0][0] * im_info[0][1]
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1
        roi_level = torch.log(torch.sqrt(h * w) / 224.0) / np.log(2)
        roi_level = torch.floor(roi_level + 4)
        # --------
        # roi_level = torch.log(torch.sqrt(h * w) / 224.0)
        # roi_level = torch.round(roi_level + 4)
        # ------
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        # roi_level.fill_(5)
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            # NOTE: need to add pyrmaid
            grid_xy = _affine_grid_gen(rois, feat_maps.size()[2:], self.grid_size)  ##
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            roi_pool_feat = self.RCNN_roi_crop(feat_maps, Variable(grid_yx).detach()) ##
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                roi_pool_feat = F.max_pool2d(roi_pool_feat, 2, 2)

        elif cfg.POOLING_MODE == 'align':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l], scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]

        elif cfg.POOLING_MODE == 'pool':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l], scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]
            
        return roi_pool_feat


    def cls_uncertainty_loss(self, pred, label, var_cls_epi_exp):
        label_gt = torch.zeros_like(pred)
        label_gt = label_gt.scatter_(1, label.unsqueeze(1), 1)

        label_other = torch.ones_like(pred)
        label_other = label_other.scatter_(1, label.unsqueeze(1), 0)

#        print(var_cls_epi_exp[0,:], label_gt[0,:], label_other[0,:])
#        assert 1==0
        return ((-1) * var_cls_epi_exp * label_gt * torch.log(pred + 1e-10) + (-1) * var_cls_epi_exp * label_other * torch.log((1-pred) + 1e-10)).sum(1)

    def cross_entropy_impl(self, pred, label): #, uncer_loc, uncer_cls):
        hypothesis = F.softmax(pred)
        label_onehot = torch.zeros_like(hypothesis)
        label_onehot.scatter_(1, label.unsqueeze(1), 1)
        print(pred, label)
        assert 1==0
#        uncer_cls_weights = torch.clamp(torch.log(1+ uncer_cls.detach()/(uncer_loc.detach() + 1e-3) + uncer_loc.detach()/(uncer_cls.detach()+1e-3)), max=2)
        #print(uncer_cls_weights, label_onehot)
        #assert 1==0
        return ((-1) * uncer_cls_weights * label_onehot * torch.log(hypothesis)).sum(dim=1).mean()

    def cross_entropy_impl2(self, pred, label):
        hypothesis = F.softmax(pred)
        label_onehot = torch.zeros_like(hypothesis)
        label_onehot.scatter_(1, label.unsqueeze(1), 1)
        #uncer_cls_weights = torch.clamp(torch.log(1+ uncer_cls.detach()/(uncer_loc.detach() + 1e-3) + uncer_loc.detach()/(uncer_cls.detach()+1e-3)), max=2)
        #print(uncer_cls_weights, label_onehot)
        #assert 1==0
        return ((-1) * label_onehot * torch.log(hypothesis)).sum(dim=1).mean()

    def gaussian(self, ins, cls_uncertain, mean=0, stddev=1, is_training=True):
        if is_training:
            noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
#            print(ins)
            return ins + cls_uncertain * noise
        return ins

    def forward(self, im_data, im_info, gt_boxes, num_boxes, epoch=0):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # Bottom-up
        c1 = self.RCNN_layer0(im_data)
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)
        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)

        p6 = self.maxpool2d(p5)

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feature_maps, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()
            pos_id = rois_label.nonzero().squeeze()
            gt_assign_pos = gt_assign[pos_id]
            rois_label_pos = rois_label[pos_id]
            rois_label_pos_ids = pos_id

            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)
            rois_label = Variable(rois_label)

            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois_label = None
            gt_assign = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois.view(-1, 5)
            pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)

        pooled_feat = self._head_to_tail(roi_pool_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        ############# Estimate Epistemic Uncertainty #############
        mean_cls_epi = []
        mean2_cls_epi = []
        mean_loc_epi = []
        mean2_loc_epi = []
        if self.training:
            iteration = 20
        else:
            iteration = 1

        for ii in range(iteration):
            roi_pool_feat_epi = F.dropout(roi_pool_feat, p=0.5, training=True)
            pooled_feat_epi = self._head_to_tail_dropout(roi_pool_feat_epi.detach())
            pooled_feat_epi = F.dropout(pooled_feat_epi, p=0.5, training=True)
            bbox_pred_epi = self.RCNN_bbox_pred(pooled_feat_epi.detach())

            cls_score_epi = self.RCNN_cls_score(pooled_feat_epi.detach())
            cls_prob_epi = F.softmax(cls_score_epi)
            mean_cls_epi.append(cls_prob_epi ** 2)
            mean2_cls_epi.append(cls_prob_epi)
            mean_loc_epi.append(bbox_pred_epi ** 2)
            mean2_loc_epi.append(bbox_pred_epi)

        mean1s_cls_epi = torch.stack(mean_cls_epi, dim=0).mean(dim=0)
        mean2s_cls_epi = torch.stack(mean2_cls_epi, dim=0).mean(dim=0)
        mean1s_loc_epi = torch.stack(mean_loc_epi, dim=0).mean(dim=0)
        mean2s_loc_epi = torch.stack(mean2_loc_epi, dim=0).mean(dim=0)

        var_cls_epi = mean1s_cls_epi - mean2s_cls_epi ** 2
        var_loc_epi = mean1s_loc_epi - mean2s_loc_epi ** 2
        var_cls_epi = var_cls_epi / var_cls_epi.max()

        if self.training:
            label_pos = torch.zeros_like(var_cls_epi)
            label_pos = label_pos.scatter_(1, rois_label.unsqueeze(1), 1) * torch.exp(var_cls_epi)

            label_neg = torch.ones_like(var_cls_epi) * torch.exp(var_cls_epi) - label_pos.max(1)[0].unsqueeze(1).repeat(1,var_cls_epi.shape[-1])

            var_cls_epi_exp = (label_pos + label_neg).detach()
            var_loc_epi_exp = torch.exp(var_loc_epi).detach()
        else:
            var_cls_epi = (var_cls_epi / var_cls_epi.max()).mean(1)
            var_loc_epi = (var_loc_epi / var_loc_epi.max()).mean(1)
#        print(var_cls_epi_exp[0,:])

#        var_cls_epi = (var_cls_epi / var_cls_epi.max())#.mean(1)
#        var_loc_epi = (var_loc_epi / var_loc_epi.max())#.mean(1)
#        ############################################################
        if self.training == False:
            #var_cls_epi = var_cls_epi / var_cls_epi.max()
            var_loc_epi = var_loc_epi / var_loc_epi.max()

        if self.training:
#            RCNN_loss_bbox = _smooth_l1_loss_w_uncertainty(bbox_pred, rois_target, loc_uncertain, rois_inside_ws, rois_outside_ws)
            RCNN_loss_bbox = _smooth_l1_loss_epi(bbox_pred, rois_target, var_loc_epi_exp, rois_inside_ws, rois_outside_ws)
#            Entropy = (-1) * (F.softmax(cls_score, dim=1) * F.log_softmax(cls_score, dim=1)).sum(1)
#            Cross_Entropy = F.cross_entropy(cls_score, rois_label, size_average=False, reduce=False)
#
#            RCNN_loss_cls = ((1-var_cls_epi) * Cross_Entropy + var_cls_epi * (-1) * Entropy).mean()

#            print (cls_prob)
            RCNN_loss_cls = self.cls_uncertainty_loss(cls_prob, rois_label, var_cls_epi_exp).mean()

        rois = rois.view(batch_size, -1, rois.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        return rois, cls_prob, var_loc_epi, var_cls_epi, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
