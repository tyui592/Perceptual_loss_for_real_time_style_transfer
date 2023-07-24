Perceptual Losses for Real-Time Style Transfer
---
**Unofficial PyTorch implementation of real-time style transfer**

**Reference**: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution, ECCV2016](https://arxiv.org/abs/1603.08155)


Requirements
--
* Pytorch (version >= 0.4.0)
* Pillow

Download
--
* The trained models can be downloaded throuth the [Google drive](https://drive.google.com/drive/folders/1_FjrtNgVGgstMFRIY6K_Fp3w1K96Zpn5?usp=sharing).
* [MSCOCO train2014](http://cocodataset.org/#download) is needed to train the network.

Usage
--

### Arguments

* `--train-flag`: Flag for train or evaluate transform network
* `--train-content`: Path of content image dataset (MSCOCO is needed)
* `--train-style`: Path of a target style image 
* `--test-content`: Path of a test content image
* `--model-load-path`: Path of trained transform network to stylize the `--test-content` image

### Train example script

```
python main.py --train-flag True --cuda-device-no 0 --imsize 256 --cropsize 240 --train-content ./coco2014/ --train-style imgs/style/mondrian.jpg --save-path trained_models/
```

### Test example script

```
python main.py --train-flag False --cuda-device-no 0 --imsize 256 --model-load-path trained_models/transform_network.pth --test-content imgs/content/chicago.jpg --output stylized.png
```

Results
--


![test_result](https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer/blob/master/imgs/figure1.png)

