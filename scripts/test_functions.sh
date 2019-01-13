GPU_ID=0
DATASET_ROOT=/home/ankush/kv/DIM/dataset/alpamatting/low_res
LOG_ROOT=/home/ankush/kv/DIM/log

cd ../tests
python test_utils.py \
   --dataset_root ${DATASET_ROOT} \
   --gpu_id ${GPU_ID} \
   --log_root ${LOG_ROOT}
