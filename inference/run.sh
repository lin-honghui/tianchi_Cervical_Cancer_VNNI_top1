# # 查看物理CPU个数
echo "[CPU nums]:"
cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l

# # 查看每个物理CPU中core的个数(即核数)
echo "[physical Core nums per CPU]:"
cat /proc/cpuinfo| grep "cpu cores"| uniq

# # 查看逻辑CPU的个数
echo "[logical Core nums per CPU]:"
cat /proc/cpuinfo| grep "processor"| wc -l

# #查看CPU信息（型号）
echo "[CPU INFO]:"
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c

# #查看内存信息
# echo "[MEM INFO]:"
# cat /proc/meminfo

date
MODEL_DIR=./model
# MODEL_DIR=../model/3_scale_rotate
# MODEL_DIR=../model/2_batchsize_2
# MODEL_DIR=../model/2_batchsize_4
# MODEL_DIR=../model/2_batchsize_8
# MODEL_DIR=../model/5_RandomShiftGtBBox


# IMAGE_DIR=../calibrate_data/Tianchi/2_dataset/VOCdevkit/VOC2007/JPEGImages
# IMAGE_DIR=/home/Liangkaihuan/tianchi/sample/data
# IMAGE_DIR=../val/image
IMAGE_DIR=/tcdata/data
RESULT_DIR=result
LIB_DIR=lib



python multi_requests_batch_inference.py -i ${IMAGE_DIR} \
                                        -r ${RESULT_DIR} \
                                        -m ${MODEL_DIR}/retinanet_i8.xml \
                                        -l ${LIB_DIR}/libcpu_extension_avx512.so \
                                        --patch_size 800 800 \
                                        --stride 800 800 \
                                        # --voc_res_file ../pred_pos.txt

cd ${RESULT_DIR}
zip -q result.zip ./*
mv result.zip ../

date



