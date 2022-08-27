import copy
from torch.nn import init
import torch
from torch import nn
import torch.nn.functional as F
import random
import math
from .osnet import osnet_x1_0, OSBlock
# from .attention import BatchDrop, BatchRandomErasing, PAM_Module, CAM_Module, SE_Module, Dual_Module
from .attention import BatchDrop, BatchFeatureErase_Top, PAM_Module, CAM_Module, SE_Module, Dual_Module
from .bnneck import BNNeck, BNNeck3
from torchvision.models.resnet import resnet50, Bottleneck
from torch.autograd import Variable
from torchvision import models

# add fuse,not weight fuse ;; only global
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


class PD_net(nn.Module):
    def __init__(self, class_num=3, droprate=0.5, stride=2, circle=False, ibn=False):
        super(PD_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 2:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, 3, droprate, return_f=False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        # print(x.size(0), x.size(1))
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class P_ag(nn.Module):
    def __init__(self, args):
        super(P_ag, self).__init__()
        #  osenet + stand/sit/lying module

        # backone
        # self.backone = ResNet(last_stride=1,
        #                       block=Bottleneck,
        #                       layers=[3, 4, 6, 3])
        # model_path_glo = './pretrained_model/resnet50-19c8e357.pth'
        # self.backone.load_param(model_path_glo)
        # print('resnet50 Loading pretrained ImageNet model ...')
        # '''
        osnet = osnet_x1_0(pretrained=True)

        self.backone = nn.Sequential(
            osnet.conv1,
            osnet.maxpool,
            osnet.conv2,
            osnet.conv3[0]
        )

        conv3 = osnet.conv3[1:]

        self.stand_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.sit_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.lying_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        # '''

        # # pose detect
        # self.pose_det = PD_net()
        # pth = 'model/dog_pose_dete_2.pth'
        # self.pose_det.load_state_dict((torch.load(pth)))
        # print('PD_net Loading pretrained ImageNet model ...')
        # for key, value in self.pose_det.named_parameters():
        #     value.requires_grad = False
        # print('PD module 参数冻结...')

        # local pooling module
        self.stand_pooling = nn.AdaptiveMaxPool2d((2, 2))
        self.sit_pooling = nn.AdaptiveMaxPool2d((3, 1))
        self.lying_pooling = nn.AdaptiveMaxPool2d((1, 3))

        self.glob_m_pooling = nn.AdaptiveMaxPool2d(1)
        self.glob_a_pooling = nn.AdaptiveAvgPool2d(1)
        # self.glob_pooling = nn.AdaptiveAvgPool2d(1)

        self.feats = 512
        reduction = BNNeck3(512, args.num_classes,
                            self.feats, return_f=True)

        self.reduction_glo = copy.deepcopy(reduction)

        # self.reduction_glo_drop = copy.deepcopy(reduction)
        # self.reduction_global = copy.deepcopy(reduction)

        self.reduction_st_global = copy.deepcopy(reduction)
        self.reduction_st_local_1 = copy.deepcopy(reduction)
        self.reduction_st_local_2 = copy.deepcopy(reduction)
        self.reduction_st_local_3 = copy.deepcopy(reduction)
        self.reduction_st_local_4 = copy.deepcopy(reduction)

        self.reduction_si_global = copy.deepcopy(reduction)
        self.reduction_si_local_1 = copy.deepcopy(reduction)
        self.reduction_si_local_2 = copy.deepcopy(reduction)
        self.reduction_si_local_3 = copy.deepcopy(reduction)

        self.reduction_ly_global = copy.deepcopy(reduction)
        self.reduction_ly_local_1 = copy.deepcopy(reduction)
        self.reduction_ly_local_2 = copy.deepcopy(reduction)
        self.reduction_ly_local_3 = copy.deepcopy(reduction)

        self.reduction_fuse = copy.deepcopy(reduction)

        # self.reduction_ch_0 = BNNeck(
        #             args.feats, args.num_classes, return_f=True)

        # self.batch_drop_block = BatchFeatureErase_Top(2048, Bottleneck)

        # 1x1 conv
        self.shared = nn.Sequential(nn.Conv2d(
            4096, 512, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        self.weights_init_kaiming(self.shared)
        self.shared_st = copy.deepcopy(self.shared)
        self.shared_sit = copy.deepcopy(self.shared)
        self.shared_ly = copy.deepcopy(self.shared)

        '''
        ...

        '''

    def forward(self, x):

        '''获取姿态参数'''
        # predict_abc = self.pose_det(x)
        # predict_abc = torch.softmax(predict_abc, dim=1)
        # A, B, C = predict_abc[:, 0], predict_abc[:, 1], predict_abc[:, 2]

        '''
        x = self.backone(x)

        glo = self.global_branch(x)
        par = self.partial_branch(x)
        cha = self.channel_branch(x)
        '''

        #
        '''
        # x = self.backone(x)
        # b, c, h, w = x.size()
        # global add batch drop
        # glo_drop, glo = self.batch_drop_block(x)
        '''
        '''特征提取'''
        x = self.backone(x)
        b, c, h, w = x.size()
        stand_feature = self.stand_branch(x)
        sit_feature = self.sit_branch(x)
        lying_feature = self.lying_branch(x)

        '''
        # # 全局特征
        # glo_feat = self.glob_a_pooling(glo)
        # glo_drop_feat = self.glob_m_pooling(glo_drop)
        # 
        # g_feat = self.reduction_glo(glo_feat)
        # g_drop_feat = self.reduction_glo_drop(glo_drop_feat)
        # p0 = p_par[:, :, 0:1, :]
        # # 局部特征
        # local_f = self.shared(x)
        # st_local_feat = self.stand_pooling(local_f)
        # si_local_feat = self.sit_pooling(local_f)
        # ly_local_feat = self.lying_pooling(local_f)
        '''

        '''stand'''
        # stand global feat 全局特征
        global_st = self.glob_a_pooling(stand_feature)
        f_st_g = self.reduction_st_global(global_st)
        # stand local feat
        # local_st = self.stand_pooling(stand_feature)
        # local_st_0 = local_st[:, :, 0:1, 0:1]
        # f_st_0 = self.reduction_st_local_1(local_st_0)
        # local_st_1 = local_st[:, :, 0:1, 1:2]
        # f_st_1 = self.reduction_st_local_2(local_st_1)
        # local_st_2 = local_st[:, :, 1:2, 0:1]
        # f_st_2 = self.reduction_st_local_3(local_st_2)
        # local_st_3 = local_st[:, :, 1:2, 1:2]
        # f_st_3 = self.reduction_st_local_4(local_st_3)

        '''st_part = {}
        for i in range(2):
            st_part[i] = local_st[:, :, 0, i]
        for i in range(2):
            st_part[i + 2] = local_st[:, :, 1, i]

        st_local_feat = torch.cat(
            (st_part[0], st_part[1], st_part[2], st_part[3]), dim=1)
        st_local_feat = st_local_feat.view(b, st_local_feat.shape[1], 1, 1)'''

        '''sit'''
        # sit global feat 全局特征
        global_sit = self.glob_a_pooling(sit_feature)
        f_si_g = self.reduction_si_global(global_sit)
        # sit local feat
        # local_si = self.sit_pooling(sit_feature)
        # local_sit_0 = local_si[:, :, 0:1, :]
        # f_sit_0 = self.reduction_si_local_1(local_sit_0)
        # local_sit_1 = local_si[:, :, 1:2, :]
        # f_sit_1 = self.reduction_si_local_2(local_sit_1)
        # local_sit_2 = local_si[:, :, 2:3, :]
        # f_sit_2 = self.reduction_si_local_3(local_sit_2)

        '''sit_part = {}
        for i in range(3):
            sit_part[i] = local_st[:, :, i, :]

        si_local_feat = torch.cat(
            (sit_part[0], sit_part[1], sit_part[2]),
            dim=1)
        si_local_feat = si_local_feat.view(b, si_local_feat.shape[1], 1, 1)
        '''

        '''lying'''
        # lying global feat 全局特征
        global_ly = self.glob_a_pooling(lying_feature)
        f_ly_g = self.reduction_ly_global(global_ly)
        # lying local feat
        # local_ly = self.lying_pooling(lying_feature)
        # local_ly_0 = local_ly[:, :, :, 0:1]
        # f_ly_0 = self.reduction_ly_local_1(local_ly_0)
        # local_ly_1 = local_ly[:, :, :, 1:2]
        # f_ly_1 = self.reduction_ly_local_2(local_ly_1)
        # local_ly_2 = local_ly[:, :, :, 2:3]
        # f_ly_2 = self.reduction_ly_local_3(local_ly_2)

        '''ly_part = {}
        for i in range(3):
            ly_part[i] = local_ly[:, :, :, i]
        ly_local_feat = torch.cat(
            (ly_part[0], ly_part[1], ly_part[2]), dim=1)
        ly_local_feat = ly_local_feat.view(b, ly_local_feat.shape[1], 1, 1)'''

        '''
        局部和全局特征 
        # stand + global
        st_g_l_feature = global_st + st_local_feat
        st_g_l_feat = self.reduction_st(st_g_l_feature)
        # sit + global
        si_g_l_feature = global_sit + si_local_feat
        si_g_l_feat = self.reduction_si(si_g_l_feature)
        # sit + global
        ly_g_l_feature = global_ly + ly_local_feat
        ly_g_l_feat = self.reduction_ly(ly_g_l_feature)
        # st_g_l_feature = st_g_l_feature.view(st_g_l_feature.size()[0], -1)
        # si_g_l_feature = si_g_l_feature.view(si_g_l_feature.size()[0], -1)
        # ly_g_l_feature = ly_g_l_feature.view(ly_g_l_feature.size()[0], -1)
        
        '''
        # 特征融合 a*st + b*si + c*ly
        st_g_l_feature = global_st.view(global_st.size()[0], -1)
        si_g_l_feature = global_sit.view(global_sit.size()[0], -1)
        ly_g_l_feature = global_ly.view(global_ly.size()[0], -1)

        # 利用全局特征进行加权融合
        # g_l_feature = C.unsqueeze(0).T * st_g_l_feature + B.unsqueeze(0).T * si_g_l_feature + A.unsqueeze(
        #     0).T * ly_g_l_feature

        # 全局和局部特征结合进行add融合
        g_l_feature = st_g_l_feature + si_g_l_feature + ly_g_l_feature

        g_l_feature = g_l_feature.view(b, g_l_feature.shape[1], 1, 1)
        fuse_feat = self.reduction_fuse(g_l_feature)


        tri_feat = [f_st_g[-1], f_si_g[-1], f_ly_g[-1]]

        if not self.training:
            return torch.stack([f_st_g[0], f_si_g[0], f_ly_g[0]], dim=2)
        # add fuse
        # return [f_st_g[1], f_st_0[1], f_st_1[1], f_st_2[1], f_st_3[1], f_si_g[1], f_sit_0[1], f_sit_1[1], f_sit_2[1],
        #         f_ly_g[1], f_ly_0[1], f_ly_1[1], f_ly_2[1], fuse_feat[1]], tri_feat

        # tree global barnch, without local brance
        return [f_st_g[1], f_si_g[1], f_ly_g[1], fuse_feat[1]], tri_feat

        # weight fuse
        # return [f_st_g[1], f_st_0[1], f_st_1[1], f_st_2[1], f_st_3[1], f_si_g[1], f_sit_0[1], f_sit_1[1], f_sit_2[1],
        #         f_ly_g[1], f_ly_0[1], f_ly_1[1], f_ly_2[1], fuse_feat[1]], predict_abc, tri_feat

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    import argparse

    parser = argparse.ArgumentParser(description='MGN')
    parser.add_argument('--num_classes', type=int, default=751, help='')
    parser.add_argument('--bnneck', type=bool, default=True)
    parser.add_argument('--pool', type=str, default='max')
    parser.add_argument('--feats', type=int, default=512)
    parser.add_argument('--drop_block', type=bool, default=True)
    parser.add_argument('--w_ratio', type=float, default=1.0, help='')

    args = parser.parse_args()
    net = MCMP_n(args)
    # net.classifier = nn.Sequential()
    # print([p for p in net.parameters()])
    # a=filter(lambda p: p.requires_grad, net.parameters())
    # print(a)

    print(net)
    input = Variable(torch.FloatTensor(8, 3, 384, 128))
    net.eval()
    output = net(input)
    print(output.shape)
    print('net output size:')
    # print(len(output))
    # for k in output[0]:
    #     print(k.shape)
    # for k in output[1]:
    #     print(k.shape)
