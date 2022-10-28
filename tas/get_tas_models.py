##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
from os import path as osp
# our modules
from tas.configure_utils import dict2config


def get_cifar_models(config):
    print(config)

    super_type = getattr(config, 'super_type', 'basic')
    # if super_type == 'basic':
    #     if config.arch == 'resnet':
    #         return CifarResNet(config.module, config.depth, config.class_num, config.zero_init_residual)
    #     elif config.arch == 'wideresnet':
    #         return CifarWideResNet(config.depth, config.wide_factor, config.class_num, config.dropout)
    #     else:
    #         raise ValueError('invalid module type : {:}'.format(config.arch))
    if super_type.startswith('infer'):
        from tas.InferCifarResNet import InferCifarResNet
        assert len(super_type.split('-')) == 2, 'invalid super_type : {:}'.format(super_type)
        infer_mode = super_type.split('-')[1]
        # if infer_mode == 'width':
        #     return InferWidthCifarResNet(config.module, config.depth, config.xchannels, config.class_num,
        #                                  config.zero_init_residual)
        # elif infer_mode == 'depth':
        #     return InferDepthCifarResNet(config.module, config.depth, config.xblocks, config.class_num,
        #                                  config.zero_init_residual)
        if infer_mode == 'shape':
            return InferCifarResNet(config.module, config.depth, config.xblocks, config.xchannels, config.class_num,
                                    config.zero_init_residual)
        else:
            raise ValueError('invalid infer-mode : {:}'.format(infer_mode))
    else:
        raise ValueError('invalid super-type : {:}'.format(super_type))


def get_imagenet_models(config):
    super_type = getattr(config, 'super_type', 'basic')
    # NAS searched architecture
    if super_type.startswith('infer'):
        assert len(super_type.split('-')) == 2, 'invalid super_type : {:}'.format(super_type)
        infer_mode = super_type.split('-')[1]
        if infer_mode == 'shape':
            from tas.InferImagenetResNet import InferImagenetResNet
            if config.arch == 'resnet':
                return InferImagenetResNet(config.block_name, config.layers, config.xblocks, config.xchannels,
                                           config.deep_stem, config.class_num, config.zero_init_residual)
            # elif config.arch == "MobileNetV2":
            #     return InferMobileNetV2(config.class_num, config.xchannels, config.xblocks, config.dropout)
            else:
                raise ValueError('invalid arch-mode : {:}'.format(config.arch))
        else:
            raise ValueError('invalid infer-mode : {:}'.format(infer_mode))
    else:
        raise ValueError('invalid super-type : {:}'.format(super_type))


def obtain_model(config):
    if config.dataset == 'cifar':
        return get_cifar_models(config)
    elif config.dataset == 'imagenet':
        return get_imagenet_models(config)
    # elif config.dataset == 'mnist' or config.dataset == 'fashion':
    #   from .CifarResNet      import GrayResNet
    #   return GrayResNet(config.module, config.depth, config.class_num, config.zero_init_residual)
    else:
        raise ValueError('invalid dataset in the model config : {:}'.format(config))


def load_net_from_checkpoint(checkpoint):
    assert osp.isfile(checkpoint), 'checkpoint {:} does not exist'.format(checkpoint)
    checkpoint = torch.load(checkpoint)
    model_config = dict2config(checkpoint['model-config'], None)
    model = obtain_model(model_config)
    model.load_state_dict(checkpoint['base-model'])
    return model
