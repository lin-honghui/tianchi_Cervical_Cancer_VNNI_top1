WORK_DIR=work_dirs/cervical_cancer/retinanet_mobilenetv2_fpn_1x/11_crop1600_libraAnchor
IMAGE_DIR=/home/Liangkaihuan/openvino/openvino_2019.3.376/deployment_tools/exps/retinanet/calibrate_data/Tianchi/2_dataset/VOCdevkit/VOC2007/JPEGImages


CUDA_VISIBLE_DEVICES=2 \
python mytools/export_to_onnx.py \
        configs/cervical_cancer/retinanet_mobilenetv2_fpn_1x.py \
        ${WORK_DIR}/latest.pth \
        --onnx_save_path=${WORK_DIR}/retinanet.onnx \
        --image_path=${IMAGE_DIR}/val_0.jpg \
        --output_names 'cls_out' 'loc_out'