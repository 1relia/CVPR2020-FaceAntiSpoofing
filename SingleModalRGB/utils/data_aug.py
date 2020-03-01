import torch


class ColorAugmentation(object):
    def __init__(self):
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor

import torch
import torch.backends.cudnn as cudnn
from torch.autograd.variable import Variable
import models
class GetFeature(object):
    def __init__(self,args,config):
        model = models.__dict__[args.stage1_arch](**config['stage1_model'])
        self.device = torch.device('cuda:' + str(args.gpus[0]) if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.random_seed)
        # args.gpus = [int(i) for i in args.gpus.split(',')]
        self.model = torch.nn.DataParallel(model,device_ids=args.gpus)
        self.model.to(self.device)
        checkpoint_dir = '/home/pengzhang/code/Sequences-FaceAntiSpoofing/SingleModalRGB/' + args.stage2_checkpoint
        checkpoint = torch.load(checkpoint_dir)
        # args.start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        self.model.load_state_dict(checkpoint['state_dict'])
        
    def __call__(self,tensor):
        self.model.eval()
        with torch.no_grad():
            input_var = Variable(tensor).float().to(self.device)
            fea = self.model(input_var)
        return fea

    #checkpoints/feathernetB-RGB@2_colortrans/_23_best.pth.tar