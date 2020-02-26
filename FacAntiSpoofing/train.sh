# CUDA_VISIBLE_DEVICES=5 nohup python main.py --config="cfgs/FeatherNetB-32.yaml" --mode RGB --save-path checkpoints/feathernetB-RGB@3 >> FeatherNetB-train-RGB@3.log &

# CUDA_VISIBLE_DEVICES=5 nohup python main.py --config="cfgs/FeatherNetB-32.yaml" --mode Depth --save-path checkpoints/feathernetB-Depth@3 >> FeatherNetB-train-Depth@3.log &

# CUDA_VISIBLE_DEVICES=4 nohup python main.py --config="cfgs/FeatherNetB-32.yaml" --mode IR --save-path checkpoints/feathernetB-IR@3 >> FeatherNetB-train-IR@3.log &

# cat FeatherNetB-train-IR@2.log | grep EER

# 2019.12.23
# # single test
# # # IR
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode IR --val True --resume checkpoints/feathernetB-IR/_50.pth.tar --val-save True --sub-prot-train 1 --sub-prot-val 1 
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode IR --val True --resume checkpoints/feathernetB-IR@2/_9.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 2
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode IR --val True --resume checkpoints/feathernetB-IR@3/_9.pth.tar --val-save True --sub-prot-train 3 --sub-prot-val 3

# # # Depth
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode Depth --val True --resume checkpoints/feathernetB-Depth/_105.pth.tar --val-save True --sub-prot-train 1 --sub-prot-val 1 
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode Depth --val True --resume checkpoints/feathernetB-Depth@2/_50.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 2
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode Depth --val True --resume checkpoints/feathernetB-Depth@3/_55.pth.tar --val-save True --sub-prot-train 3 --sub-prot-val 3
# # # RGB
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode RGB --val True --resume checkpoints/feathernetB-RGB/_105.pth.tar --val-save True --sub-prot-train 1 --sub-prot-val 1 
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode RGB --val True --resume checkpoints/feathernetB-RGB@2/_50.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 2
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode RGB --val True --resume checkpoints/feathernetB-RGB@3/_46.pth.tar --val-save True --sub-prot-train 3 --sub-prot-val 3


# 2019.12.23
# # cross test(跨人种) Protocl 4 Train and Val
# # sub-prot 4_1 4_2 4_3
# #   train   A    C   E
# #    val    A    C   E
# #   test    C&E  A&E A&C
# # A: Africa, C: Central Asia, E: East Asia
# # 4@12_dev.txt == A&C 4@23_dev.txt == C&E 4@13_dev.txt == A&E
# # --sub-prot-val 23 == 4@23_dev.txt == C&E
# # IR
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode IR --val True  --resume checkpoints/feathernetB-IR/_50.pth.tar --val-save True --sub-prot-train 1 --sub-prot-val 23 
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode IR --val True --resume checkpoints/feathernetB-IR@2/_9.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 13
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode IR --val True --resume checkpoints/feathernetB-IR@3/_9.pth.tar --val-save True --sub-prot-train 3 --sub-prot-val 12

# # # Depth
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode Depth --val True --resume checkpoints/feathernetB-Depth/_105.pth.tar --val-save True --sub-prot-train 1 --sub-prot-val 23 
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode Depth --val True --resume checkpoints/feathernetB-Depth@2/_50.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 13
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode Depth --val True --resume checkpoints/feathernetB-Depth@3/_55.pth.tar --val-save True --sub-prot-train 3 --sub-prot-val 12
# # # RGB
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode RGB --val True --resume checkpoints/feathernetB-RGB/_105.pth.tar --val-save True --sub-prot-train 1 --sub-prot-val 23 
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode RGB --val True --resume checkpoints/feathernetB-RGB@2/_50.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 13
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode RGB --val True --resume checkpoints/feathernetB-RGB@3/_46.pth.tar --val-save True --sub-prot-train 3 --sub-prot-val 12

# 2019.12.24
# # 用非洲人训练，在中亚和东亚人上进行分别测试，看实验结果,看看东亚和中亚在跨模态数据上的差异
# # train 2/3 val 1
# # IR
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode IR --val True --resume checkpoints/feathernetB-IR@2/_9.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 1
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode IR --val True --resume checkpoints/feathernetB-IR@3/_9.pth.tar --val-save True --sub-prot-train 3 --sub-prot-val 1
# # Depth
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode Depth --val True --resume checkpoints/feathernetB-Depth@2/_50.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 1
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode Depth --val True --resume checkpoints/feathernetB-Depth@3/_55.pth.tar --val-save True --sub-prot-train 3 --sub-prot-val 1
# # RGB
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode RGB --val True --resume checkpoints/feathernetB-RGB@2/_50.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 1
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode RGB --val True --resume checkpoints/feathernetB-RGB@3/_46.pth.tar --val-save True --sub-prot-train 3 --sub-prot-val 1

# 2019.12.24
# 中亚和东亚跨模态进行测试 2->C 3->E
# IR
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode IR --val True --resume checkpoints/feathernetB-IR@2/_9.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 3
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode IR --val True --resume checkpoints/feathernetB-IR@3/_9.pth.tar --val-save True --sub-prot-train 3 --sub-prot-val 2
# # RGB
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode RGB --val True --resume checkpoints/feathernetB-RGB@2/_50.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 3
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetB-32.yaml" --mode RGB --val True --resume checkpoints/feathernetB-RGB@3/_46.pth.tar --val-save True --sub-prot-train 3 --sub-prot-val 2


# 2019.12.25
    
# RGB cross modal training 
# 因为经过上面的跨域测试发现，train 2(C) val 1(A) 效果最差，所以希望可以将这个跨域做好，其他的情况应该也会变好。

# (1) 在第一个卷积中使用FRN，第一个倒置残差卷积中的第一个卷积后使用Norm_Activation（norm_type）
# CUDA_VISIBLE_DEVICES=7 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode RGB --save-path checkpoints/feathernetB-RGB@2_frn --sub-prot-train 2 --sub-prot-val 1  >> FeatherNetB-train-RGB@2_frn.log &

# CUDA_VISIBLE_DEVICES=7 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode RGB --save-path checkpoints/feathernetB-RGB@2_sw2 --sub-prot-train 2 --sub-prot-val 1 >> FeatherNetB-train-RGB@2_sw2.log &

# CUDA_VISIBLE_DEVICES=2 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode RGB --save-path checkpoints/feathernetB-RGB@2_sw3 --sub-prot-train 2 --sub-prot-val 1 >> FeatherNetB-train-RGB@2_sw3.log &

# CUDA_VISIBLE_DEVICES=3 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode RGB --save-path checkpoints/feathernetB-RGB@2_sw5 --sub-prot-train 2 --sub-prot-val 1 >> FeatherNetB-train-RGB@2_sw5.log &

# CUDA_VISIBLE_DEVICES=3 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode RGB --save-path checkpoints/feathernetB-RGB@2_in --sub-prot-train 2 --sub-prot-val 1 >> FeatherNetB-train-RGB@2_in.log &

# CUDA_VISIBLE_DEVICES=4 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode RGB --save-path checkpoints/feathernetB-RGB@2_bn --sub-prot-train 2 --sub-prot-val 1 >> FeatherNetB-train-RGB@2_bn.log &

# 




# 2019.12.26
# (2) 在第一个卷积中使用FRN，第一个倒置残差卷积中的前两个卷积都使用Norm_Activation（norm_type）
# CUDA_VISIBLE_DEVICES=7 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode RGB --save-path checkpoints/feathernetB-RGB@2_frn2 --sub-prot-train 2 --sub-prot-val 1  >> FeatherNetB-train-RGB@2_frn2.log &

# CUDA_VISIBLE_DEVICES=4 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode RGB --save-path checkpoints/feathernetB-RGB@2_sw2_2 --sub-prot-train 2 --sub-prot-val 1 >> FeatherNetB-train-RGB@2_sw2_2.log &

# test 
# CUDA_VISIBLE_DEVICES=4 python main.py --config="cfgs/FeatherNetNorm.yaml" --mode RGB --val True --resume checkpoints/feathernetB-RGB@2_frn/_39_best.pth.tar --val-save True --sub-prot-train 2 --sub-prot-val 13

# 2019.12.27
# train
# 通过上面的实验发现（1）中的frn在跨模态上实验效果最好，所以在此测试一下C->A IR上的实验效果 
# CUDA_VISIBLE_DEVICES=7 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode IR --save-path checkpoints/feathernetB-IR@2_frn --sub-prot-train 2 --sub-prot-val 1  >> FeatherNetB-train-IR@2_frn.log &
# 修改 FeatherNetNorm.yaml 的norm_type为sw2
# CUDA_VISIBLE_DEVICES=6 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode IR --save-path checkpoints/feathernetB-IR@2_sw2 --sub-prot-train 2 --sub-prot-val 1  >> FeatherNetB-train-IR@2_sw2.log &
# 修改 FeatherNetNorm.yaml 的norm_type为sw3
# CUDA_VISIBLE_DEVICES=4 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode IR --save-path checkpoints/feathernetB-IR@2_sw3 --sub-prot-train 2 --sub-prot-val 1  >> FeatherNetB-train-IR@2_sw3.log &
# 修改 FeatherNetNorm.yaml 的norm_type为sw5
# CUDA_VISIBLE_DEVICES=5 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode IR --save-path checkpoints/feathernetB-IR@2_sw5 --sub-prot-train 2 --sub-prot-val 1  >> FeatherNetB-train-IR@2_sw5.log &
# 修改 FeatherNetNorm.yaml 的norm_type为bn
# CUDA_VISIBLE_DEVICES=1 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode IR --save-path checkpoints/feathernetB-IR@2_bn --sub-prot-train 2 --sub-prot-val 1  >> FeatherNetB-train-IR@2_bn.log &
# 修改 FeatherNetNorm.yaml 的norm_type为in
# CUDA_VISIBLE_DEVICES=5 nohup python main.py --config="cfgs/FeatherNetNorm.yaml"  --mode IR --save-path checkpoints/feathernetB-IR@2_in --sub-prot-train 2 --sub-prot-val 1  >> FeatherNetB-train-IR@2_in.log &
