#!/bin/bash
ROOT=/research/pheng4/qdliu/hzyang/;
DATAROOT="/research/pheng4/qdliu/hzyang/seg/dataset/"
BATCH_SIZE=2;AUG=0;
results_dir="$ROOT/oct/atta/exps/oct_a{$AUG}b{$BATCH_SIZE}"
export CUDA_VISIBLE_DEVICES="$1"
echo "test OCT on GPU $1"
mkdir -p "$results_dir"
cp -rf "$0" "$results_dir"
python3 train.py \
--tepochs=1 \
--alr=1e-3 \
--task=seg_oct \
--batch-size=1 \
--td=1,64,64,64,64,11 \
--img_path="$DATAROOT"/spectralis/hctrain/image/ \
--label_path="$DATAROOT"/spectralis/hctrain/label/ \
--vimg_path="$DATAROOT"/cirrus/image/ \
--vlabel_path="$DATAROOT"/cirrus/label/ \
--sub_name="$DATAROOT"/cirrus/cirrus_name.txt \
--img_ext=png \
--label_ext=txt \
--results_dir="$results_dir"/ \
--ss=1 \
--resume_T=$results_dir/checkpoints/tnet_checkpoint.pth \
--resume_AE=$results_dir/checkpoints/aenet_checkpoint.pth \
--wo=1 \
--wt=1,0,1,1,1,1 \
--seq=1,2,3 \
--si \
-t
