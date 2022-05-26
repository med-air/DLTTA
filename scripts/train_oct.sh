#!/bin/bash
ROOT=/research/pheng4/qdliu/hzyang/;
TRAINER=$2; BATCH_SIZE=2;AUG=0;SEGLOSS='ce';
DATAROOT="/research/pheng4/qdliu/hzyang/seg/dataset/"
results_dir="$ROOT/oct/atta/exps/oct_a{$AUG}b{$BATCH_SIZE}"
export CUDA_VISIBLE_DEVICES="$1"
echo "train OCT $TRAINER on GPU $1"
mkdir -p "$results_dir"
cp -rf "$0" "$results_dir"
if [ $TRAINER == "tnet" ]; then
    python3 train.py \
    --epochs=20 \
    --task=seg_oct \
    --segloss=$SEGLOSS \
    --batch-size=$BATCH_SIZE \
    --td=1,64,64,64,64,11 \
    --aangle=0,0 \
    --aprob=$AUG \
    --img_path="$DATAROOT"/spectralis/hctrain/image/ \
    --label_path="$DATAROOT"/spectralis/hctrain/label/ \
    --vimg_path="$DATAROOT"/spectralis/hcval/image/ \
    --vlabel_path="$DATAROOT"/spectralis/hcval/label/ \
    --trainer=$TRAINER \
    --img_ext=png \
    --label_ext=txt \
    --results_dir="$results_dir"
else
    python3 train.py \
    --epochs=20 \
    --task=seg_oct \
    --segloss=$SEGLOSS \
    --batch-size=$BATCH_SIZE \
    --td=1,64,64,64,64,11 \
    --aangle=0,0 \
    --aprob=0 \
    --img_path="$DATAROOT"/spectralis/hctrain/image/ \
    --label_path="$DATAROOT"/spectralis/hctrain/label/ \
    --vimg_path="$DATAROOT"/spectralis/hcval/image/ \
    --vlabel_path="$DATAROOT"/spectralis/hcval/label/ \
    --trainer=$TRAINER \
    --img_ext=png \
    --label_ext=txt \
    --segaeloss=mse \
    --results_dir="$results_dir"/ \
    --wt=1,0,1,1,1,1,1 \
    --resume_T=$results_dir/checkpoints/tnet_checkpoint_e20.pth
fi
