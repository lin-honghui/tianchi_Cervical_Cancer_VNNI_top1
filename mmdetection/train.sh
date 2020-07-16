# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_dwhead_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_dwhead_1x/0_baseline

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_dwhead_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_dwhead_1x/1_addBN

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_dwhead_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_dwhead_1x/2_addBN


# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/0_baseline

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/1_addBN

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/2_crop960_resize800
# crop_size=960

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/3_crop1280_resize800
# crop_size=1280

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/4_fpn128
# crop_size=800

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/5_fpn128_crop1280_resize800
# crop_size=1280

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/7_fpn128_head64_crop1280_resize800
# crop_size=1280

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/8_fpn128_head64_crop1600_resize800
# crop_size=1600

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/9_fpn128_head64_crop1440_resize800
# crop_size=1440

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/10_crop1600_freezeBN
# crop_size=1600

CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/11_crop1600_libraAnchor
crop_size=1600

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/12_crop1600_balanceL1
# crop_size=1600

# CONFIG=configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py
# WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/13_crop1600_libraAnchor_balanceL1
# crop_size=1600

GPU_ID=2
## train
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/train.py ${CONFIG}
## test on val imgs
sh mytools/test.sh ${WORK_DIR} ${CONFIG} ${GPU_ID} ${crop_size}
