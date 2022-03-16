KITTI2015(){
cd ..
cd playground

KITTI=2015
datapath=/datagrid01/liu/kitti_stereo/kitti_2015/testing/
savepath=kitti2015_prediction
model=NGNet
maxdisp=192
devices=0
seed=1
loadmodel=/home/zliu/Desktop/new_test/Backups/model_best.pth
norm_path=/home/zliu/Desktop/new_test/Backups/scripts/model_best.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python kitti_submission.py \
                    --KITTI $KITTI \
                    --datapath $datapath \
                    --savepath $savepath \
                    --model $model \
                    --maxdisp $maxdisp \
                    --devices $devices \
                    --seed $seed \
                    --loadmodel $loadmodel \
                    --norm_path $norm_path
}

KITTI2015