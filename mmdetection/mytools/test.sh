WORK_DIR=$1
CONFIG=$2
GPU_ID=$3
crop_size=$4
DATA_DIR=data/Cervical_Cancer/val

# CUDA_VISIBLE_DEVICES=${GPU_ID} \
# python mytools/test.py ${CONFIG} \
#                         ${WORK_DIR}/latest.pth \
#                         --image_dir=${DATA_DIR}/image \
#                         --result_dir=${WORK_DIR}/result \
#                         --voc_res_file=${WORK_DIR}/voc_pred_pos.txt \
#                         --patch_size 800 800 \
#                         --strides 800 800 \
                        # --vis \
                        # --vis_image_dir=${WORK_DIR}/vis_image \
                        # --score_thr=0.3 \
                        # --ann_dir=${DATA_DIR}/label


CUDA_VISIBLE_DEVICES=${GPU_ID} \
python mytools/test.py ${CONFIG} \
                        ${WORK_DIR}/epoch_90.pth \
                        --image_dir=${DATA_DIR}/image \
                        --result_dir=${WORK_DIR}/result \
                        --voc_res_file=${WORK_DIR}/voc_pred_pos.txt \
                        --patch_size ${crop_size} ${crop_size} \
                        --strides ${crop_size} ${crop_size}