Semantic segmentation with dynamic upsamplers, based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

For example, to train UperNet-R50 with [CARAFE](https://github.com/myownskyW7/CARAFE) in FPN:

```shell
bash dist_train.sh configs/dynamic_upsampling/upernet_r50_4xb4_carafe-80k_ade20k-512x512.py 4
```
We find that the performance on ADE20K is unstable and may fluctuate in (-0.5, +0.5) mIoU.
