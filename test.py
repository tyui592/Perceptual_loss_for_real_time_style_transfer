import torch

from network import TransformNetwork
from image_utils import imload, imsave

def load_transform_network(args):
    transform_network = TransformNetwork()
    transform_network.load_state_dict(torch.load(args.model_load_path))
    return transform_network

def network_test(args):
    device = torch.device("cuda" if args.cuda_device_no >= 0 else 'cpu')

    transform_network = load_transform_network(args)
    transform_network = transform_network.to(device)

    input_image = imload(args.test_content, args.imsize).to(device)

    with torch.no_grad():
        output_image = transform_network(input_image)

    imsave(output_image, args.output)

    return None
