# CVPR2020-FaceAntiSpoofing
Main code of CVPR2020 Chalearn Single-modal (RGB) Cross-ethnicity Face anti-spoofing Recognition Challenge@CVPR2020

3DResnet for face expression detect
FaceAntiSpoofing use Feathernet to identify true or false

## 3D Resnet

change the train model and config in cfgs/3DResnet.yaml<br>
3 protocols, need 3 commands

### Train
```
CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/ResNet3D.yaml" \
--mode RGB --save-path checkpoints/ResNet3D5-RGB@1 \
--sub-prot-train 1 --sub-prot-val 1

CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/ResNet3D.yaml" \
--mode RGB --save-path checkpoints/ResNet3D5-RGB@2 \
--sub-prot-train 2 --sub-prot-val 2

CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/ResNet3D.yaml" \
--mode RGB --save-path checkpoints/ResNet3D5-RGB@3 \
--sub-prot-train 3 --sub-prot-val 3
```
### dev
```
CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/ResNet3D.yaml" \
    --mode RGB --val True --phase-test True \    
    --data-root /home/pengzhang/dataset/CASIA-CeFA/phase1/ \
    --resume checkpoints/ResNet3D5-RGB@1/1_best.pth.tar
    --val-save True --sub-prot-train 1 --sub-prot-test 1
```
### test
```
CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/ResNet3D.yaml" \
    --mode RGB --val True --phase-test True \    
    --data-root /home/pengzhang/dataset/CASIA-CeFA/phase1/ \
    --resume checkpoints/ResNet3D5-RGB@1/1_best.pth.tar
    --val-save True --sub-prot-train 1 --sub-prot-test 1
 ```
### merge
```
cd submission/
python merge_file_for_final_submission.py
```

## FaceAntiSpoofing
change the train model and config in cfgs/FeatherNetNorm.yaml<br>
3 protocols, need 3 commands

### Train
```
CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/FeatherNetNorm.yaml" \
--mode RGB --save-path checkpoints/FeatherNetNorm-RGB@1 \
--sub-prot-train 1 --sub-prot-val 1

CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/FeatherNetNorm.yaml" \
--mode RGB --save-path checkpoints/FeatherNetNorm-RGB@2 \
--sub-prot-train 2 --sub-prot-val 2

CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/FeatherNetNorm.yaml" \
--mode RGB --save-path checkpoints/FeatherNetNorm-RGB@3 \
--sub-prot-train 3 --sub-prot-val 3
```
### dev
```
CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/FeatherNetNorm.yaml" \
    --mode RGB --val True --phase-test True \    
    --data-root /home/pengzhang/dataset/CASIA-CeFA/phase1/ \
    --resume checkpoints/FeatherNetNorm-RGB@1/1_best.pth.tar
    --val-save True --sub-prot-train 1 --sub-prot-test 1
```
### test
```
CUDA_VISIBLE_DEVICES=0 python main.py --config="cfgs/FeatherNetNorm.yaml" \
    --mode RGB --val True --phase-test True \    
    --data-root /home/pengzhang/dataset/CASIA-CeFA/phase1/ \
    --resume checkpoints/FeatherNetNorm-RGB@1/1_best.pth.tar
    --val-save True --sub-prot-train 1 --sub-prot-test 1
 ```
### merge
```
cd submission/
python merge_file_for_final_submission.py
```
## submit
get the 3Dresnet_submission.txt and Feathernet_submission.txt in 3DResnet/submission/
```
cp /FaceAntiSpoofing/submission/submission.txt /3Dresnet/submission/
python fusion_submission.py
```
