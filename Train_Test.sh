#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python trainval_net.py ResNet101 --dataset pascal_voc --cuda --mGPUs --bs 16 --cag --r True --checkepoch 7 --checkpoint 625
CUDA_VISIBLE_DEVICES=0 python trainval_net.py ResNet101 --dataset pascal_voc --cuda --bs 2 --cag

#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset pascal_voc_0712 --cuda --cag --checkepoch 6 --checkpoint 8274
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset pascal_voc --cuda --cag --checkepoch 7 --checkpoint 625 #8274
#CUDA_VISIBLE_DEVICES=5 python test_net.py ResNet101 --dataset pascal_voc --cuda --cag --checkepoch 8 --checkpoint 625 #8274


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python trainval_net.py ResNet101 --dataset pascal_voc_0712 --cuda --mGPUs --bs 16 --cag
#CUDA_VISIBLE_DEVICES=0 python trainval_net.py ResNet101 --dataset pascal_voc_0712 --cuda --bs 1 --cag

#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset pascal_voc_0712 --cuda --cag --checkepoch 6 --checkpoint 8274
#CUDA_VISIBLE_DEVICES=5 python test_net.py ResNet101 --dataset pascal_voc_0712 --cuda --cag --checkepoch 8 --checkpoint 2067 #8274
#CUDA_VISIBLE_DEVICES=5 python test_net.py ResNet101 --dataset pascal_voc_0712 --cuda --cag --checkepoch 9 --checkpoint 2067 #8274

#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet101 --dataset coco --cuda --mGPUs --bs 8 --cag --r True --checkepoch 2 --checkpoint 29315
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --cuda --cag --checkepoch 6 --checkpoint 2504
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --cuda --cag --checkepoch 7 --checkpoint 2504
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --cuda --cag --checkepoch 8 --checkpoint 2504
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset coco --cuda --cag --checkepoch 9 --checkpoint 29315
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --cuda --cag --checkepoch 10 --checkpoint 1251

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python trainval_net.py ResNet101 --dataset pascal_voc_0712 --cuda --mGPUs --bs 16 --cag --r True --checkepoch 1 --checkpoint 2067
#CUDA_VISIBLE_DEVICES=0 python trainval_net.py ResNet101 --dataset pascal_voc_0712 --cuda --bs 1 --cag

#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset pascal_voc_0712 --cuda --cag --checkepoch 7 --checkpoint 2067 #8274
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset pascal_voc_0712 --cuda --cag --checkepoch 8 --checkpoint 2067 #8274
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset pascal_voc_0712 --cuda --cag --checkepoch 9 --checkpoint 8274
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset pascal_voc_0712 --cuda --cag --checkepoch 10 --checkpoint 4136



###      COCO     ###
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet101 --dataset coco --cuda --mGPUs --bs 8 --cag --s 2 --r True --checksession 2 --checkepoch 2 --checkpoint 29315 #--lr 1e-
#CUDA_VISIBLE_DEVICES=2 python trainval_net.py ResNet101 --dataset coco --cuda --bs 1 --cag --s 2 #--r True --checksession 2 --checkepoch 2 --checkpoint 29315 #--lr 1e-

#CUDA_VISIBLE_DEVICES=2 python test_net.py ResNet101 --dataset coco --cuda --cag --checksession 2 --checkepoch 9 --checkpoint 29315 --vis


### Visual Genome ###
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet101_0.01 --dataset vg --cuda --mGPUs --bs 8 --cag
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101_0.01 --dataset vg --cuda --cag --checkepoch 9 --checkpoint 21985 #10992


## KITTI
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet101 --dataset kitti --cuda --mGPUs --bs 8 --cag
#CUDA_VISIBLE_DEVICES=0 python trainval_net.py ResNet101 --dataset kitti --cuda --bs 2 --cag

#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset kitti --cuda --cag --checkepoch 11 --checkpoint 918
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset kitti --cuda --cag --checkepoch 12 --checkpoint 918
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset kitti --cuda --cag --checkepoch 13 --checkpoint 918
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset kitti --cuda --cag --checkepoch 14 --checkpoint 918
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101 --dataset kitti --cuda --cag --checkepoch 15 --checkpoint 918

#CUDA_VISIBLE_DEVICES=0 python trainval_net.py ResNet101 --dataset kitti --cuda --bs 1 --cag

