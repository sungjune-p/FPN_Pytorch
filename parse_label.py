import glob
import shutil
import os
import numpy as np
import argparse
import subprocess
from subprocess import PIPE, STDOUT
try:
    from subprocess import DEVNULL # py3k
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

def eval_cpp():
    num_test_images = 3799
    file_path = os.path.dirname(os.path.realpath(__file__))
    # print file_path
    # assert 1==0
    gt_labels     = glob.glob('./cpp/gt_label/*.txt')
    eval_labels   = glob.glob('./cpp/pred_label_tmp/*.txt')


    dir_gt     = './cpp/gt_label' # os.path.join(file_path,'gt_label')
    dir_eval   = './cpp/pred_label_tmp' #os.path.join(file_path,'eval')

    print 'number of labels'
    print 'gt labels   :' + str(len(gt_labels))
    print 'eval        :' + str(len(eval_labels))

    #assert len(eval_labels) == num_test_images
    without_ext_eval   = [os.path.basename(file_name) for file_name in eval_labels]

    without_ext_gt = [os.path.basename(file_name) for file_name in  gt_labels]

    for file_name in without_ext_eval:
        assert file_name in without_ext_eval

    val_label_path     = './cpp/gt_label/' #os.path.join(file_path,'cpp/gt_label')
    eval_parsed_path   = './cpp/pred_label/plot' #os.path.join(file_path,'cpp/pred_label')


    parsed_paths = [val_label_path,eval_parsed_path]
    # if os.path.exists(eval_parsed_path) :
    #     raise ValueError('backup your previous evaluate results')

    for path in parsed_paths:
        if not os.path.exists(path):
            os.makedirs(path)


    for i,path in enumerate(np.sort(without_ext_eval)):
        base_new = str(i).rjust(6,'0')+'.txt'


        dst_eval   = os.path.join(eval_parsed_path , base_new)
        dst_gt     = os.path.join(val_label_path,base_new)

        src_eval   = os.path.join(dir_eval,path)
        src_gt     = os.path.join(dir_gt,path)

        shutil.copy(src_eval,dst_eval)
        shutil.copy(src_gt,dst_gt)

    bashCommands = ["g++ evaluate_object.cpp -o evaluate_object","./evaluate_object"]
    for bashCommand in bashCommands:
        process = subprocess.Popen(bashCommand.split(), stdin=PIPE, stdout=DEVNULL, stderr=STDOUT,cwd='./cpp')
        output, error = process.communicate()

    result_files = ['./cpp/pred_label/plot/car_detection.txt',
                     './cpp/pred_label/plot/cyclist_detection.txt',
                     './cpp/pred_label/plot/pedestrian_detection.txt']
    object_list = ['car','cyclist','pedestrian']
    difficulties = ['easy','moderate','hard']
    means = []

    for i,result_file in enumerate(result_files):
        assert os.path.isfile(result_file)

        f = open(result_file,'r')
        lines = f.readlines()
        lines_arr = []
        # print lines
        for line in lines:
            lines_arr.append(np.array(line.strip().split()).astype(np.float))

        lines_arr = np.array(lines_arr)
        print (lines_arr)
        print object_list[i]
        print np.mean(np.array(lines_arr[:,1:]),axis=0)
        means.append(np.mean(np.array(lines_arr[:,1:]),axis=0))
    # print means

    return np.array(means)
if __name__ == '__main__':
	mean = eval_cpp()
	print mean
