# 得到最后test的提交文件

# 提交的文件格式，先放单个协议的dev，接着放对应的test，如下：
# dev/003000 0.15361   #Note:  line 1- the first row of 4@1_dev_res.txt
#             ......
# test/000001 0.94860   #Note:  line 201- the first row of 4@1_test_res.txt           
#             ......
# dev/003200 0.40134   #Note:  line 2401- the first row of  4@2_dev_res.txt     
#             ......   
# test/001201 0.23847   #Note:  line 2601- the first row of  4@2_test_res.txt
#             ......
# dev/003400 0.23394   #Note:  line 4801- the first row of  4@3_dev_res.txt    
#             ......
# test/001201 0.62544   #Note:  line 5001- the first row of  4@3_test_res.txt  

# # 所以先需要得到dev中的结果，
# # 运行命令,注意选择合适的配置，模态，子协议，
# CUDA_VISIBLE_DEVICES=1 python main.py --config="cfgs/FeatherNetNorm.yaml" \
#     --mode RGB \
#     --val True --phase-test True \
#     --data-root /home/pengzhang/dataset/CASIA-CeFA/phase1/ \
#     --resume checkpoints/feathernetB-RGB@2_colortrans/_23_best.pth.tar \
#     --val-save True \
#     --sub-prot-train 2 --sub-prot-test 1

# CUDA_VISIBLE_DEVICES=1 python main.py --config="cfgs/FeatherNetNorm.yaml" \
#     --mode RGB \
#     --val True --phase-test True \
#     --data-root /home/pengzhang/dataset/CASIA-CeFA/phase1/ \
#     --resume checkpoints/feathernetB-RGB@2_colortrans/_23_best.pth.tar \
#     --val-save True \
#     --sub-prot-train 2 --sub-prot-test 2

# CUDA_VISIBLE_DEVICES=1 python main.py --config="cfgs/FeatherNetNorm.yaml" \
#     --mode RGB \
#     --val True --phase-test True \
#     --data-root /home/pengzhang/dataset/CASIA-CeFA/phase1/ \
#     --resume checkpoints/feathernetB-RGB@2_colortrans/_23_best.pth.tar \
#     --val-save True \
#     --sub-prot-train 2 --sub-prot-test 3

# # 然后再得到test的结果，
# CUDA_VISIBLE_DEVICES=1 python main.py --config="cfgs/FeatherNetNorm.yaml" \
#     --mode RGB \
#     --val True --phase-test True \
#     --data-root /home/pengzhang/dataset/CASIA-CeFA/phase2/ \
#     --resume checkpoints/feathernetB-RGB@2_colortrans/_23_best.pth.tar \
#     --val-save True \
#     --sub-prot-train 2 --sub-prot-test 1

# CUDA_VISIBLE_DEVICES=1 python main.py --config="cfgs/FeatherNetNorm.yaml" \
#     --mode RGB \
#     --val True --phase-test True \
#     --data-root /home/pengzhang/dataset/CASIA-CeFA/phase2/ \
#     --resume checkpoints/feathernetB-RGB@2_colortrans/_23_best.pth.tar \
#     --val-save True \
#     --sub-prot-train 2 --sub-prot-test 2

# CUDA_VISIBLE_DEVICES=1 python main.py --config="cfgs/FeatherNetNorm.yaml" \
#     --mode RGB \
#     --val True --phase-test True \
#     --data-root /home/pengzhang/dataset/CASIA-CeFA/phase2/ \
#     --resume checkpoints/feathernetB-RGB@2_colortrans/_23_best.pth.tar \
#     --val-save True \
#     --sub-prot-train 2 --sub-prot-test 3

# 最后将所有的结果进行合并，排列顺序为dev1 test1  dev2 test2 ...
cd submission
python merge_file_for_finalsubmission.py
