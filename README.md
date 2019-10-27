Perceptl Losses for Real-Time Style Transfer
---
**Unofficial PyTorch implementation of real-time style transfer**

**Reference**: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution, ECCV2016](https://arxiv.org/abs/1603.08155)

**Contact**: `Minseong Kim` (tyui592@gmail.com)

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

### Train example script

```
python main.py --train-flag True --cuda-device-no 0 --imsize 256 --cropsize 240 --train-content-image-path ./coco2014/ --train-style-image-path sample_images/style_images/mondrian.jpg --save-path trained_models/
```

### Test example script

```
python main.py --train-flag False --cuda-device-no 0 --imsize 256 --model-load-path trained_models/network.pth --test-image-path sample_images/content_images/chicago.jpg --output-image-path stylized.png
```

Example Images
--

* Content image: sample_images/content_images/chicago.jpg
* Style image: sample_images/style_images/mondrian.jpg

![test_result](https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer/blob/master/sample_images/test_results/chicago_mondrian.png)

* Content image: smaple_images/content_images/chicago.jpg
* Style image: sample_images/style_images/abstraction.jpg

![test_result2](https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer/blob/master/sample_images/test_results/chicago_abstraction.png)
