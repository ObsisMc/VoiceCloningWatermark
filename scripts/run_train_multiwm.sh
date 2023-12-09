#!/bin/bash

beta=1
lam=10000
alpha=10000
gamma=0
dtw=false
lr=0.00001
val_itvl=500
val_size=50
num_epochs=50
batch_size=8
experiment=1
cuda_n=3
summary="multi_preatk_gradatk_WMwl8lr1e-5audioMSElam1e4alpha1e4gamma0"
transform="WM"
num_points=16000
n_fft=1000
hop_length=400
num_layers=4
watermark_len=8
shift_ratio=0
from_checkpoint="1-multi_IDwl8lr1e-4audioMSElam100/50-1-multi_IDwl8lr1e-4audioMSElam100.pt"
share_param=false
permutation=false
stft_small=true
ft_container="mag"
thet=1
mp_encoder="single"
mp_decoder="unet"
mp_join="mean"
embed="stretch"
luma=false
ROOT=/home/rz60/codes/watermarking

if [ -d "$ROOT/outputs/$experiment-$summary" ]; then
    while true; do
        read -p "Folder $experiment-$summary already exists. Do you wish to overwrite?" yn
        case $yn in
            [Yy]* ) rm -rf "$ROOT/outputs/$experiment-$summary"; break;;
            [Nn]* ) echo "Rename your experiment. Exiting... "; exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
fi
mkdir -p "$ROOT/outputs/$experiment-$summary"
mkdir -p "$ROOT/logs/$experiment-$summary"

cat > "$ROOT/outputs/$experiment-$summary/parameters.txt" <<EOF
===Hyperparameters===:
lr: $lr
batch_size: $batch_size

===Loss func hyperparameters===:
beta: $beta
theta: $thet
DTW lambda: $lam (disregard if not using DTW)

===Architecture hyperparameters===:
Using permutation? $permutation
Transform: $transform
Using small bottleneck? $stft_small
What ft container? $ft_container
Embedding style: $embed
mp_encoder: $mp_encoder
mp_decoder: $mp_decoder
mp_join: $mp_join
Using luma? $luma

===Training parameters===:
Epochs: $num_epochs

===Validation parameters===:
val_itvl: $val_itvl its
val_size: $val_size 

CUDA_VISIBLE_DEVICES=X,Y python3 $ROOT/src/main.py --beta $beta --lam $lam --dtw $dtw --lr $lr --val_itvl $val_itvl --val_size $val_size --num_epochs $num_epochs --batch_size $batch_size --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --permutation $permutation --transform $transform --stft_small $stft_small --ft_container $ft_container --thet $thet --mp_encoder $mp_encoder --mp_decoder $mp_decoder --mp_join $mp_join --embed $embed --luma $luma --num_points $num_points --n_fft $n_fft --hop_length $hop_length --num_layers $num_layers --watermark_len $watermark_len --shift_ratio $shift_ratio --alpha $alpha --gamma $gamma --share_param $share_param
EOF

CUDA_VISIBLE_DEVICES=$cuda_n python3 $ROOT/src/main.py --beta $beta --lam $lam --dtw $dtw --lr $lr --val_itvl $val_itvl --val_size $val_size --num_epochs $num_epochs --batch_size $batch_size --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --permutation $permutation --transform $transform --stft_small $stft_small --ft_container $ft_container --thet $thet --mp_encoder $mp_encoder --mp_decoder $mp_decoder --mp_join $mp_join --embed $embed --luma $luma --num_points $num_points --n_fft $n_fft --hop_length $hop_length --num_layers $num_layers --watermark_len $watermark_len --shift_ratio $shift_ratio --alpha $alpha --gamma $gamma --share_param $share_param
