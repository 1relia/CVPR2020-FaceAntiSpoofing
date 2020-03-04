
from glob import glob 

txt_files = sorted(glob('*submission_*.txt'))
print(len(txt_files))
txt_files = txt_files[-6:] 
for i in txt_files:
    print(i)

final_submission_file = 'submission_feathernetB-RGB_w15d15.txt'
# 确保包含dev 1,2,3和test 1,2,3
# RGB@1_submission_dev.txt
# RGB@2_submission_dev.txt
# RGB@3_submission_dev.txt
# RGB@1_submission_test.txt
# RGB@2_submission_test.txt
# RGB@3_submission_test.txt
split_ = [txt.split('_')[3] for txt in txt_files]
print(split_)
assert split_[0][-1] == '1' and split_[1][-1] == '2' and split_[2][-1] == '3'
assert split_[3][-1] == '1' and split_[4][-1] == '2' and split_[5][-1] == '3'  
f = open(final_submission_file,'w')

# 写入文本
# 排列顺序为dev1 test1  dev2 test2 ...
for i in range(3):
    lines = open(txt_files[i],'r').readlines()
    for line in lines:
        f.write(line)
    lines = open(txt_files[i+3],'r').readlines()
    for line in lines:
        f.write(line)
print('----done-----')


