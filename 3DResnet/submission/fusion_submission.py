from glob import glob

resnet_submission = '3Dresnet@2_submission.txt'

feathernet_submission = 'submission2.txt'

final_submission_file = 'final_submission_file.txt'

f = open(final_submission_file, 'w')

length = len(open(feathernet_submission, 'r').readlines())

resnet_lines = open(resnet_submission, 'r').readlines()
feathernet_lines = open(feathernet_submission, 'r').readlines()

print(length)

for i in range(length):
    resnet_line = resnet_lines[i]
    feathernet_line = feathernet_lines[i]
    dir_name = resnet_line.split(' ')[0]
    resnet_result = resnet_line.split(' ')[1]
    resnet_result.strip()
    feathernet_result = feathernet_line.split(' ')[1]
    feathernet_result.strip()
    if float(resnet_result) < 0.35 and float(feathernet_result) > 0.9 and 'test' in dir_name:
        f.write(dir_name + ' ' + resnet_result)
    else:
        f.write(dir_name + ' ' + feathernet_result)
    print(dir_name, resnet_result, feathernet_result)

