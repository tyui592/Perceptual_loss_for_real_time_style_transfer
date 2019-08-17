import torch

from network import TransformNetwork
from image_utils import imload, imsave

def load_transform_network(args):
    transform_network = TransformNetwork()
    transform_network = transform_network.load_state_dict(torch.load(args.model_load_path))
    return transform_network

def network_test(args):
    device = torch.device("cuda:%d"%args.cuda_device_no) if args.cuda_device_no >= 0 else torch.device('cpu')

    transform_network = load_transform_network(args.model_load_path).to(device)

    input_image = imload(args.test_content_image_path, args.imsize).to(device)

    with torch.no_grad():
        output_image = transform_network(input_image)

    imsave(output_iamge, args.output_image_path)

    return None
