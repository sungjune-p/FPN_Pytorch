# --------------------------------------------------------
# Pytorch FPN implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, based on code from faster R-CNN
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io as sio
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import time
import cv2
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms, soft_nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections
from model.utils.blob import im_list_to_blob
from model.fpn.resnet import resnet
import parse_label

import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="weights",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--soft_nms', help='whether use soft_nms', action='store_true')
    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_0712_trainval"
        args.imdbval_name = "voc_0712_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "kitti":
        args.imdb_name = "kittivoc_train"
        args.imdbval_name = "kittivoc_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "cfgs/{}.yml".format(args.net)


    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    if args.exp_name is not None:
        input_dir = args.load_dir + "/" + args.net + "/" + args.dataset + '/' + args.exp_name
    else:
        input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fpn = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fpn = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fpn = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fpn = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fpn.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fpn.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fpn.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
#    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
#                             imdb.num_classes, training=False, normalize=False)
#    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
#                                             shuffle=False, num_workers=4,
#                                             pin_memory=True)
#
#    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fpn.eval()

#    matlab_score = []
#    matlab_uncer_cls = []
#    matlab_uncer_loc = []
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    num_images = 3799
    num_classes = 4
    _classes = ['__background__', 'pedestrian', 'car', 'cyclist']
    for i in range(num_images):
        img = cv2.imread('./data/KITTIVOC/JPEGImages/' + str(i+3682).zfill(6)+'.jpg').astype('float32') - cfg.PIXEL_MEANS
        blobs, im_scales = _get_image_blob(img)

        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        #data = data_iter.next()
        #print (data[0].min())
        #assert 1==0
        #im_data.data.resize_(data[0].size()).copy_(data[0])
        #im_info.data.resize_(data[1].size()).copy_(data[1])
        #gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        #num_boxes.data.resize_(data[3].size()).copy_(data[3])

#        mean1s = []
#        mean2s = []
#
#        mean1b = []
#        mean2b = []
#        var1s = []
#        var1b = []
        det_tic = time.time()
#        for ii in range(1):
        rois, cls_prob, bbox_pred, \
        _, _, _, _, _ = fpn(im_data, im_info, gt_boxes, num_boxes)

#            mean1s.append(cls_score ** 2)
#            mean2s.append(cls_score)
#            mean1b.append(bbox_pred **2)
#            mean2b.append(bbox_pred)
#
#        mean1s_ = torch.stack(mean1s, dim=0).mean(dim=0)
#        mean2s_ = torch.stack(mean2s, dim=0).mean(dim=0)
#
#        mean1b_ = torch.stack(mean1b, dim=0).mean(dim=0)
#        mean2b_ = torch.stack(mean2b, dim=0).mean(dim=0)
#
#        var1 = mean1s_ - (mean2s_ ** 2)
#        var1_norm = var1 #/ var1.max()
#
#        var2 = mean1b_ - (mean2b_ ** 2)
#        var2_norm = var2 #/ var2.max()
#        var2_norm = torch.squeeze(var2_norm,0)

        #rois, cls_prob, cls_score, bbox_pred, \
        #_, _, _, _, _ = fpn(im_data, im_info, gt_boxes, num_boxes, th=0)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = boxes

        pred_boxes /= im_scales[0] #.cuda()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread('./data/KITTIVOC/JPEGImages/' + str(i+3682).zfill(6)+'.jpg').astype('float32')
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]

#                var_norm_cls = var1_norm[inds]
#                var_norm_loc = var2_norm[inds]
#
#                var_norm_cls = var_norm_cls[order]
#                var_norm_loc = var_norm_loc[order]
                if args.soft_nms:
                    np_dets = cls_dets.cpu().numpy().astype(np.float32)
                    keep = soft_nms(np_dets, cfg.TEST.SOFT_NMS_METHOD)  # np_dets will be changed in soft_nms
                    keep = torch.from_numpy(keep).type_as(cls_dets).int()
                    cls_dets = torch.from_numpy(np_dets).type_as(cls_dets)
                else:
                    keep = nms(cls_dets, cfg.TEST.NMS)
                
                cls_dets = cls_dets[keep.view(-1).long()]
#                var_norm_cls = var_norm_cls[keep.view(-1).long()]
#                var_norm_loc = var_norm_loc[keep.view(-1).long()]

#                matlab_score = np.concatenate([matlab_score, cls_dets[:,-1].cpu().numpy()], axis=0)
#                matlab_uncer_cls = np.concatenate([matlab_uncer_cls, var_norm_cls[:,-1].data.cpu().numpy()], axis=0)
#                matlab_uncer_loc = np.concatenate([matlab_uncer_loc, var_norm_loc[:,-1].data.cpu().numpy()], axis=0)
#                sio.savemat('np_struct_arr.mat', {'matlab_score': matlab_score, 'matlab_uncer_cls': matlab_uncer_cls, 'matlab_uncer_loc': matlab_uncer_loc})
#                assert 1==0
#                print (matlab_score.shape, matlab_uncer_cls.shape, matlab_uncer_loc.shape)
                if vis:
                    #im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), var_norm_cls.data.cpu().numpy(), var_norm_loc.data.cpu().numpy(), j, 0.7)
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.7)

                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]



        cls_output_location = './kitti_test/results/exp1/data'

        if not os.path.exists(cls_output_location):
            os.makedirs(cls_output_location)

        txt_file = os.path.join(cls_output_location, str(str(i+3682).zfill(6)) + '.txt')

        with open(txt_file, 'wt') as f:
            for j in xrange(1, imdb.num_classes):
                dets = all_boxes[j][i]
                if dets.shape[0] == 0:
                    continue
                for k in xrange(dets.shape[0]):
                    cls = _classes[j]

                    f.write('{:s} -1 -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1 -1 -1 -1 {:.4f}\n'.format( \
                            cls, dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], dets[k, 4]))

        if vis:
            cv2.imwrite('images/result%d.png' % (i), im2show)
        if i % 10 == 0:
            print ('im_detect: {:d}/{:d}'.format(i + 1, num_images))

    #parse_label.eval_cpp()
    assert 1==0
#        misc_toc = time.time()
#        nms_time = misc_toc - misc_tic
#
#        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
#                         .format(i + 1, num_images, detect_time, nms_time))
#        sys.stdout.flush()
#
#        if vis:
#            cv2.imwrite('images/result%d.png' % (i), im2show)
#            assert 1==0
            #pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

#    sio.savemat('np_struct_arr.mat', {'matlab_score': matlab_score, 'matlab_uncer_cls': matlab_uncer_cls, 'matlab_uncer_loc': matlab_uncer_loc})

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)


    print('Evaluating detections')

    results = []
    overthresh = 0.5
    imdb.evaluate_detections(all_boxes, output_dir, overthresh)
    #results.append(recall_val)
#    print('Overthresh: ', overthresh)

#    results = []
#    overthresh = np.arange(0.5, 1.0, 0.05)
#    for t in overthresh:
#        recall_val = imdb.evaluate_detections(all_boxes, output_dir, t)
#        results.append(recall_val)
#    print('Overthresh: ', overthresh)
#    print('Recall: ', results)
#    print('mean : ', sum(results) / len(results))


#    print('Evaluating detections')
#    imdb.evaluate_detections(all_boxes, output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))
