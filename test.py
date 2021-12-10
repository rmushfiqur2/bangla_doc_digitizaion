import argparse
import os

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from torch.backends import cudnn
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess
import cv2
import matplotlib.pyplot as plt


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images per batch among all devices')
    parser.add_argument('--thres', type=float, default=0.3, help='Minimum confidence level of object')
    parser.add_argument('--iou_thres', type=float, default=0.2, help='Maximum allowed intersection over union')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--imsave', action='store_true',
                        help='whether to save the output of bounding boxes. Will be saved in test/ folder ')
    parser.add_argument('--imwrite', action='store_true',
                        help='whether to save the cropped graphemes in the data_path/project/grapheme folder ')

    args = parser.parse_args()
    return args


def test(opt):
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': False,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

    threshold = opt.thres
    iou_threshold = opt.iou_thres

    use_cuda = False
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True


    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])]),
                          return_img_path=True)
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    for iter, data in enumerate(val_generator):
        with torch.no_grad():
            imgs = data['img']
            img_path = data['path']
            #if params.num_gpus == 1:
                #imgs = imgs.cuda()

            features, regression, classification, anchors = model(imgs)
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(imgs,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)
            imgs = imgs.permute(0, 2, 3, 1)
            imgs = imgs.numpy() # still mean and std dev effect present
            imgs = imgs + 0.95
            for i in range(len(imgs)):
                if len(out[i]['rois']) == 0:
                    continue
                if opt.imwrite:
                    grepheme_img = img_path[i].replace('train', 'grapheme')
                    grepheme_img = grepheme_img.replace('test', 'grapheme')
                    grepheme_img = grepheme_img.replace('cropped', 'grapheme')
                    grepheme_img_base, _ = os.path.splitext(grepheme_img)
                #img = cv2.imread(img_path[i])
                img = imgs[i] * 255
                if opt.imsave:
                    viz_img = img.copy()
                for j in range(len(out[i]['rois'])):
                    (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
                    if opt.imwrite:
                        grapheme = img[y1:y2, x1:x2]
                        # print(grapheme.shape)
                        # cv2.imwrite(grepheme_img_base+'-%03d-%03d-%03d-%03d-%03d'%(j+1,x1, y1, x2, y2)+'.jpg', grapheme)
                        print(grepheme_img_base+'-%03d-%03d-%03d-%03d-%03d'%(j+1,x1, y1, x2, y2)+'.jpg')
                    if opt.imsave:
                        # obj = obj_list[out[i]['class_ids'][j]]
                        score = float(out[i]['scores'][j])
                        ss = 0 #int(255-score*255)
                        cv2.rectangle(viz_img, (x1, y1), (x2, y2), (ss, ss, ss), 2)
                        cv2.putText(viz_img, '{:.2f}'.format(score),
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)
                if opt.imsave:
                    cv2.imwrite('test/'+os.path.split(img_path[i])[-1], viz_img)


if __name__ == '__main__':
    opt = get_args()
    test(opt)
    # python test.py -c 0 -p bangla_numbers -w "logs/bangla_numbers/efficientdet-d0_499_5500.pth" --imsave --thres 0.3
