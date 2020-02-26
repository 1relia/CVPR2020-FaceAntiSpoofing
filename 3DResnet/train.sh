CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/ResNet3D.yaml" \
--mode RGB \
--save-path checkpoints/ResNet3D10-RGB@2 \
--sub-prot-train 2 --sub-prot-val 1


# CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/ResNet3D.yaml" \
# --mode RGB \
# --save-path checkpoints/ResNet3D5readdata2_bs60-RGB@2 \
# --sub-prot-train 2 --sub-prot-val 1 \
# --val True \
# --resume checkpoints/ResNet3D5readdata2_bs60-RGB@2/1_best.pth.tar

# test
# CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/ResNet3D.yaml" \
#     --mode RGB \
#     --val True --phase-test True \
#     --data-root /home/pengzhang/dataset/CASIA-CeFA/phase1/ \
#     --resume checkpoints/ResNet3D5readdata2_bs60-RGB@2/1_best.pth.tar
#     --val-save True \
#     --sub-prot-train 2 --sub-prot-test 1
