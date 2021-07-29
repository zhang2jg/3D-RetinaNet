#!/bin/bash
cwd=$(pwd)
mkdir DL
home_dir=$cwd/DL
cd $home_dir

# git clone repos
git clone # https://github.com/gurkirt/3D-RetinaNet.git
git clone https://github.com/gurkirt/road-dataset.git

# install dependencies
#pip3 install torch torchvision torchaudio tensorboardx
yes | conda create -n dl
conda activate dl
yes | conda install pytorch torchvision torchaudio tensorboardx scipy cudatoolkit=10.2 -c pytorch
#conda install -c conda-forge tensorboardx

# process videos
road_dir="road"
mkdir $road_dir
cd $road_dir
#bash ../road-dataset/road/get_dataset.sh
sftp -P 15022  18018626116:pRixN49a@pan.blockelite.cn <<END
get /CloudData/DL_data/instance_counts.json .
get /CloudData/DL_data/road_trainval_v1.0.json .
get -r /CloudData/DL_data/videos .
get -r /CloudData/DL_data/kinetics-pt ${home_dir}
END

#sftp 18018626116:pRixN49a@pan.blockelite.cn:15022/CloudData/DL_data/videos .
#sftp 18018626116:pRixN49a@pan.blockelite.cn:15022/CloudData/DL_data/instance_counts.json .
#sftp 18018626116:pRixN49a@pan.blockelite.cn:15022/CloudData/DL_data/road_trainval_v1.0.json .

cd $home_dir
yes | sudo apt install ffmpeg
python ./road-dataset/extract_videos2jpgs.py ./road/

# download kinetics-pt
#cd 3D-RetinaNet/kinetics-pt
#bash ./get_kinetics_weights.sh
#cd $home_dir
#sftp -P 15022  18018626116:pRixN49a@pan.blockelite.cn <<END
#get -r /CloudData/DL_data/kinetics-pt .
#END

# run command (train and gen_det)
cd $home_dir/3D-RetinaNet
# train -
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py ${home_dir}/ ${home_dir}/ ${home_dir}/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041
#CUDA_VISIBLE_DEVICES=0 python main.py ${home_dir}/ ${home_dir}/ ${home_dir}/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=1 --LR=0.0041

# test and building tubes
python main.py ${home_dir}/ ${home_dir}/ ${home_dir}/kinetics-pt/ --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041

# upload results to cloud
today=$(date +%Y-%m-%d_%H-%M-%S)
result_file=result-${today}.txt
cp screenlog.0 ${result_file}
sftp -P 15022  18018626116:pRixN49a@pan.blockelite.cn <<END
put ${result_file} /CloudData/DL_data
END

## to run validation using pre-trained model from google drive
#sftp -P 15022  18018626116:pRixN49a@pan.blockelite.cn <<END
#get /CloudData/DL_data/model_000030.pth /root/DL/road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadt3-h3x3x3
#END

## send email (https://kenfavors.com/code/how-to-install-and-configure-sendmail-on-ubuntu/). this does not work. the error is the email is blocked because of sender ip address. see file '/var/mail/root'
#sudo apt-get update
#sudo apt-get install --reinstall apache2