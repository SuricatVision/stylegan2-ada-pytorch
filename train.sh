export CUDA_VISIBLE_DEVICES=0,1
python train.py --outdir=train_logs/ --data=/home/dereyly/ImageDB/ffhq/ffhq_512.zip --gpus=2 --cfg=paper512 --resume=ffhq512 --mirror=1 #--batch=16
