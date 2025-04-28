import os

os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
  --train-file ./data/91-image_x4.h5 \
  --eval-file ./data/Set5_x4.h5 \
  --outputs-dir ./results \
  --scale 4 \
  --lr 1e-4 \
  --batch-size 16 \
  --num-epochs 60 \
  --num-workers 2 \
  --seed 42')
