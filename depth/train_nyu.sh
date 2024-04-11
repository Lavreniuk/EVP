PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1,3 python3 -m torch.distributed.launch --nproc_per_node=2 \
--use_env --master_port=25676 train.py --batch_size 2 --dataset nyudepthv2 --data_path ./ \
 --max_depth 10.0 --max_depth_eval 10.0 --weight_decay 0.1 \
 --num_filters 32 32 32 --deconv_kernels 2 2 2\
 --flip_test --shift_window_test \
 --shift_size 4 --save_model --layer_decay 0.9 --drop_path_rate 0.3 --log_dir $1 \
  --crop_h 480 --crop_w 480 --epochs 15 ${@:2} --max_lr 2e-4
