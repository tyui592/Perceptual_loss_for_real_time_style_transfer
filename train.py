import time

import torch
import torch.nn as nn
import torchvision

from network import TransformNetwork
from image_utils import ImageFolder, get_transformer, imload, imsave

mse_criterion = torch.nn.MSELoss(reduction='mean')

def extract_features(model, x, layers):
    features = list()
    for index, layer in enumerate(model):
        x = layer(x)
        if index in layers:
            features.append(x)
    return features

def calc_Content_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1/len(features)] * len(features)
    
    content_loss = 0
    for f, t, w in zip(features, targets, weights):
        content_loss += mse_criterion(f, t) * w
        
    return content_loss

def gram(x):
    b ,c, h, w = x.size()
    g = torch.bmm(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1,2))
    return g.div(h*w)

def calc_Gram_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1/len(features)] * len(features)
        
    gram_loss = 0
    for f, t, w in zip(features, targets, weights):
        gram_loss += mse_criterion(gram(f), gram(t)) * w
    return gram_loss

def calc_TV_Loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss

def network_train(args):
    device = torch.device("cuda" if args.cuda_device_no >= 0 else 'cpu')

    # Transform Network
    transform_network = TransformNetwork()
    transform_network = transform_network.to(device)
    
    # Content Data set
    train_dataset = ImageFolder(args.train_content, get_transformer(args.imsize, args.cropsize))

    # Loss network
    loss_network = torchvision.models.__dict__[args.vgg_flag](pretrained=True).features.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(params=transform_network.parameters(), lr=args.lr)

    # Target style image
    target_style_image = imload(args.train_style, imsize=args.imsize).to(device)
    b, c, h, w = target_style_image.size()
    target_style_image = target_style_image.expand(args.batchs, c, h, w)

    # Train
    loss_logs = {'content_loss':[], 'style_loss':[], 'tv_loss':[], 'total_loss':[]}
    for iteration in range(args.max_iter):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchs, shuffle=True)
        image = next(iter(train_dataloader))
        image = image.to(device)

        output_image = transform_network(image)

        target_content_features = extract_features(loss_network, image, args.content_layers)
        target_style_features = extract_features(loss_network, target_style_image, args.style_layers) 

        output_content_features = extract_features(loss_network, output_image, args.content_layers)
        output_style_features = extract_features(loss_network, output_image, args.style_layers)

        content_loss = calc_Content_Loss(output_content_features, target_content_features)
        style_loss = calc_Gram_Loss(output_style_features, target_style_features)
        tv_loss = calc_TV_Loss(output_image)
        
        total_loss = content_loss * args.content_weight + style_loss * args.style_weight + tv_loss * args.tv_weight

        loss_logs['content_loss'].append(content_loss.item())
        loss_logs['style_loss'].append(style_loss.item())
        loss_logs['tv_loss'].append(tv_loss.item())
        loss_logs['total_loss'].append(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # print loss logs
        if iteration % args.check_iter == 0:
            str_ = '%s: iteration: [%d/%d/],\t'%(time.ctime(), iteration, args.max_iter)
            for key, value in loss_logs.items():
                # check most recent 100 loss values
                str_ += '%s: %2.2f,\t'%(key, sum(value[-100:])/100)
            print(str_)
            
            imsave(output_image.cpu(), args.save_path+"training_images.png")

            torch.save(transform_network.state_dict(), args.save_path+"transform_network.pth")

    # save train results
    torch.save(loss_logs, args.save_path+"loss_logs.pth")
    torch.save(transform_network.state_dict(), args.save_path+"transform_network.pth")

    return transform_network
