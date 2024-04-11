PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 \
--use_env --master_port=25677 test.py --dataset kitti --data_path ./ \
 --max_depth 80.0 --max_depth_eval 80.0 \
 --num_filters 32 32 32 --deconv_kernels 2 2 2\
 --shift_size 20 --shift_window_test  --kitti_crop garg_crop\
 --flip_test  --save_visualize\
 --ckpt_dir $1 \
  --crop_h 352 --crop_w 352 ${@:2}
