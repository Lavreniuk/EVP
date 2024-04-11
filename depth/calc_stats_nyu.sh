PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 \
--use_env --master_port=25671 calc_stats.py --batch_size 10 --dataset nyudepthv2 --data_path ./ \
 --max_depth 10.0 --max_depth_eval 10.0 --weight_decay 0.1 \
 --num_filters 32 32 32 --deconv_kernels 2 2 2\
 --flip_test --shift_window_test \
 --shift_size 2 --save_model --layer_decay 0.9 --drop_path_rate 0.3 --log_dir $1 \
  --crop_h 480 --crop_w 480 --epochs 15 ${@:2} --max_lr 3e-4
