
mode = 'gen_dets'
import os

for net,d in [('I3D',1),('RCN',2),('C2D',3)]:
    for train_subset in ['train_1', 'train_2','train_3']:
        for seq, bs, tseqs in [(16,8, [16,32]), (8,4,[8,32])]:
            for tseq in tseqs: 
                if not (train_subset == 'train_1' and seq == 8):
                    cmd = 'CUDA_VISIBLE_DEVICES={:d} python main.py --MODE={:s} --MODEL_TYPE={:s} --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN={:d} --TRAIN_SUBSETS={:s} --SEQ_LEN={:d} --BATCH_SIZE={:d} \ &'.format(d, mode, net, tseq, train_subset, seq, bs)
                    print(cmd)
                    # os.system(cmd )