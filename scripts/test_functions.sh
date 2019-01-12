GPU_ID=0
DATASET_ROOT=/home/ankush/kv/DIM/dataset/alpamatting/low_res


cd ../utils
python loader.py --dataset_root ${DATASET_ROOT} --gpu_id ${GPU_ID}

