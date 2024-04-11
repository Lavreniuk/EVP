logdir=$2
mkdir -p $logdir

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 12346 calc_stats.py \
--dataset $1 --model_id $1 \
--batch-size 10 --lr 0.00008 --wd 1e-2 --print-freq 100 \
--epochs 20 --img_size 512 ${@:4} \
2>&1 | tee $logdir/log.txt
