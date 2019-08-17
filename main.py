import os
import argparse

from train import network_train
from test import network_test

def build_parser():
    parser = argparse.ArgumentParser()

    # cpu, gpu mode selection
    parser.add_argument('--cuda-device-no', type=int,
                    help='cpu : -1, gpu : 0 ~ n ', default=0)

    ### arguments for network training
    parser.add_argument('--train-flag', type=bool,
                    help='Train flag', default=False)

    parser.add_argument('--epochs', type=int,
                    help='Train epoch', default=2)

    parser.add_argument('--batchs', type=int,
                    help='Batch size', default=8)

    parser.add_argument('--lr', type=float,
                    help='Learning rate to optimize network', default=0.1)

    parser.add_argument('--check-iter', type=int,
                    help='Number of iteration to check training logs', default=100)

    parser.add_argument('--view-flag', type=bool,
                    help='View training logs when traing network on jupyter notebook', default=False)

    parser.add_argument('--imsize', type=int,
                    help='Size for resize image during training', default=256)

    parser.add_argument('--cropsize', type=int,
                    help='Size for crop image durning training', default=240)

    parser.add_argument('--vgg-flag', type=str,
                    help='VGG flag for calculating losses', default='vgg16')

    parser.add_argument('--content-layers', type=int, nargs='+', 
                    help='layer indices to extract content features', default=[15])
    
    parser.add_argument('--style-layers', type=int, nargs='+',
                    help='layer indices to extract style features', default=[3, 8, 15, 22])

    parser.add_argument('--content-weight', type=float, 
                    help='content loss weight', default=1.0)
    
    parser.add_argument('--style-weight', type=float,
                    help='style loss weight', default=30.0)

    parser.add_argument('--tv-weight', type=float,
                    help='tv loss weight', default=1.0)

    parser.add_argument('--train-content-image-path', type=str,
                    help='Content images path for training', default='./coco2014/')

    parser.add_argument('--train-content-image-list', type=str,
                    help='Content image lists for training', default='./coco2014/train.txt')

    parser.add_argument('--train-style-image-path', type=str,
                    help='The taget syle image path for training', default='sample_images/style_images/mondrian.jpg')

    parser.add_argument('--save-path', type=str,
                    help='Save path', default='./trained_models/')

    ### arguments for network evalution
    parser.add_argument('--model-load-path', type=str,
                    help="Trained model load path", default="./trained_models/transform_network.pth")

    # content image file name
    parser.add_argument('--test-image-path', type=str,
                    help="test content image path", default='sample_images/content_images/chicago.jpg')

    # output file name for network test
    parser.add_argument('--output-image-path', type=str,
                    help='output image path to save the stylized image', default='result_image/stylized.jpg')

    # style strengths of style images    
    parser.add_argument('--style-strength', type=float,
                    help='style strength for interpolation of multiple style images', default=1.0)
    
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args= parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_no)

    for key, val in vars(args).items():
        print(key, val)

    if args.train_flag:
        transform_network = network_train(args)
    else:
        network_test(args)
