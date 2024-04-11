PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1,3 python3 -m torch.distributed.launch --nproc_per_node=2 \
--use_env --master_port=25673 test.py --dataset nyudepthv2 --data_path ./ \
 --max_depth 10.0 --max_depth_eval 10.0 \
 --num_filters 32 32 32 --deconv_kernels 2 2 2\
 --shift_size 8 --shift_window_test \
 --flip_test --save_visualize\
 --ckpt_dir $1 \
  --crop_h 480 --crop_w 480 ${@:2}
