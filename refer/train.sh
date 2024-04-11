logdir=$2
mkdir -p $logdir

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node $3 --master_port 12346 train.py \
--dataset $1 --model_id $1 \
--batch-size 3 --lr 8e-5 --wd 1e-2 --print-freq 100 \
--epochs 20 --img_size 512 ${@:4} \
2>&1 | tee $logdir/log.txt
