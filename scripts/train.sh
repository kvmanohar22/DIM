GPU_ID=0
DATASET_ROOT=/home/ankush/kv/DIM/dataset/alpamatting/low_res
LOG_ROOT=/home/ankush/kv/DIM/log
BATCH_SIZE=2
BASE_LR=0.00001
MAX_EPOCHS=1000


CUDA_VISIBLE_DEVICES=0 python train.py \
   --train_mode \
   --gpu_id ${GPU_ID} \
   --dataset_root ${DATASET_ROOT} \
   --log_root ${LOG_ROOT} \
   --base_lr ${BASE_LR} \
   --max_epochs ${MAX_EPOCHS} \
   --batch_size ${BATCH_SIZE}
