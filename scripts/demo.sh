GPU_ID=-1

# Set these variables
CKPT_PATH=ckpt/demo.npy
IMG_PATH=imgs/demo/imgs/img0.png
TRI_PATH=imgs/demo/trimaps/tri0.png

CUDA_VISIBLE_DEVICES=0 python demo.py \
   --gpu_id ${GPU_ID} \
   --ckpt_path ${CKPT_PATH} \
   --img_path ${IMG_PATH} \
   --tri_path ${TRI_PATH}
