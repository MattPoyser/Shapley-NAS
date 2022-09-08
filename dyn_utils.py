import utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import sys
sys.path.insert(0, "/home/matt/Documents/hem/perceptual")
sys.path.insert(0, "/home2/lgfm95/hem/perceptual")
sys.path.insert(0, "C:\\Users\\Matt\\Documents\\PhD\\x11\\HEM\\perceptual")
sys.path.insert(0, "/hdd/PhD/hem/perceptual")
from dataloader_classification import DynamicDataset

def get_data(args):
    train_transform, test_transform = utils._data_transforms_general(args)
    pretrain_resume = "/home2/lgfm95/hem/perceptual/good.pth.tar"
    grayscale = False
    is_detection = False
    convert_to_paths = False
    convert_to_lbl_paths = False
    isize = 64
    nz = 8
    aisize = 3
    is_concat=False
    if args.set == "mnist":
        dset_cls = dset.MNIST
        dynamic_name = "mnist"
        grayscale = True
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercMnistGood.pth.tar"
        aisize = 1
    elif args.set == "fashion":
        dset_cls = dset.FashionMNIST
        dynamic_name = "fashion"
        grayscale = True
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercFashionGood.pth.tar"
        aisize = 1
    elif args.set == "cifar10":
        dset_cls = dset.CIFAR10
        dynamic_name = "cifar10"
        auto_resume = "/home2/lgfm95/hem/perceptual/tripletCifar10MseKGood.pth.tar"
        nz = 32
    elif args.set == "cifar100":
        dset_cls = dset.CIFAR100
        dynamic_name = "cifar100"
        auto_resume = "/home2/lgfm95/hem/perceptual/tripletCifar10MseKGood.pth.tar"
        nz = 32
    elif args.set == "imagenet":
        dynamic_name = "imagenet"
        isize = 256
        auto_resume = "/home2/lgfm95/hem/perceptual/ganPercImagenetGood.pth.tar"
        convert_to_paths = True
    else:
        raise TypeError("Unknown dataset : {:}".format(args.name))

    if args.isbad:
        auto_resume = "badpath"

    normalize = transforms.Normalize(
        mean=[0.13066051707548254],
        std=[0.30810780244715075])
    perc_transforms = transforms.Compose([
        transforms.RandomResizedCrop(isize),
        transforms.ToTensor(),
        normalize,
    ])
    if args.dynamic:
        # print(perc_transforms)
        train_data = DynamicDataset(
            perc_transforms=perc_transforms,
            pretrain_resume=pretrain_resume,
            image_transforms=train_transform,
            val_transforms=test_transform,
            val=False,
            dataset_name=dynamic_name,
            auto_resume=auto_resume,
            hardness=args.hardness,
            isize=isize,
            nz=nz,
            aisize=aisize,
            grayscale=grayscale,
            isTsne=True,
            tree=args.isTree,
            subset_size=args.subset_size,
            is_csv=args.is_csv,
            is_detection=is_detection,
            is_concat=is_concat,
            seed=args.seed)
            # is_csv=False)
        # is_csv=False)
        if args.set == "imagenet":
            test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name,
                                   subset_size=10000)
        else:
            test_data = dset_cls(root=args.data, train=False, download=False, transform=test_transform)
    else:
        if args.vanilla:
            if args.set == "imagenet":
                subset_size = 10000
                dynamic_name = "imagenet"
                train_data = SubDataset(transforms=train_transform, val_transforms=test_transform, val=False,
                                        dataset_name=dynamic_name, subset_size=subset_size)
                test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name, subset_size=subset_size)
            else:
                train_data = dset_cls(root=args.data, train=True, download=False, transform=train_transform)
                test_data = dset_cls(root=args.data, train=False, download=False, transform=test_transform)
        else: #abl
            if args.set == "imagenet":
                train_data = SubDataset(transforms=train_transform, val_transforms=test_transform, val=False,
                                        dataset_name=dynamic_name, subset_size=args.subset_size)
                test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name, subset_size=args.subset_size)
            else:
                subset_size = args.subset_size
                train_data = SubDataset(transforms=train_transform, val_transforms=test_transform, val=False, dataset_name=dynamic_name, subset_size=subset_size)
                test_data = SubDataset(transforms=test_transform, val=True, dataset_name=dynamic_name, subset_size=subset_size)

    return train_data, test_data
