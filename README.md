Semantic segmentation with dynamic upsamplers, based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

For example, to train UPerNet-R50 with [CARAFE](https://github.com/myownskyW7/CARAFE) in FPN:

```shell
bash dist_train.sh configs/dynamic_upsampling/upernet_r50_4xb4_carafe-80k_ade20k-512x512.py 4
```
We find that the performance on ADE20K is unstable and may fluctuate about (-0.5, +0.5) mIoU.

The code of upsampler application on [SegFormer](https://github.com/NVlabs/SegFormer)(Semantic Segmentation) and [DepthFormer](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox)(Monocular Depth Estimation) can be found [here](https://github.com/tiny-smart/segmentation-with-upsamplers/releases).
