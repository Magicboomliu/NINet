KITTI0(){
cd  ..
loss=config/loss_config_disp.json
outf_model=models_saved/kitti_320
logf=logs/kitti_505
datapath=/datagrid01/liu/kitti_stereo
datathread=4
lr=4e-4
devices=0
dataset=kitti
trainlist=filenames/KITTI_mix_train_psedo.txt
vallist=filenames/KITTI_mix_val_psedo.txt
startR=0
startE=0
batchSize=1
testbatch=2
maxdisp=-1
model=none
save_logdir=experiments_logdir/kitti
model=EDNet
norm_path=/home/zliu/Desktop/new_test/Backups/scripts/model_best.pth
pretrain=/home/zliu/Desktop/new_test/Backups/model_best.pth


python3 -W ignore train.py  --cuda --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --model $model \
               --maxdisp $maxdisp \
               --datapath $datapath \
               --manualSeed 1024 --test_batch $testbatch \
               --save_logdir $save_logdir \
               --norm_path $norm_path \
               --pretrain $pretrain
}


KITTI0
