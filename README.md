# **PerAct** 
## Annotated Tutorial 

**Important: Before starting, change the runtime to GPU.**

This notebook is an annotated tutorial on training [Perceiver-Actor (PerAct)](https://peract.github.io/) from scratch. We will look at training a single-task agent on the `open_drawer` task.  The tutorial will start from loading calibrated RGB-D images, and end with visualizing *action detections* in voxelized observations. Overall, this guide is 
meant to complement the [paper](https://peract.github.io/) by providing concrete implementation details.  

## Credit
This notebook heavily builds on data-loading and pre-preprocessing code from [`ARM`](https://github.com/stepjam/ARM), [`YARR`](https://github.com/stepjam/YARR), [`PyRep`](https://github.com/stepjam/PyRep), [`RLBench`](https://github.com/stepjam/RLBench) by [Stephen James et al.](https://stepjam.github.io/) The [PerceiverIO](https://arxiv.org/abs/2107.14795) code is adapted from [`perceiver-pytorch`](https://github.com/lucidrains/perceiver-pytorch) by [Phil Wang (lucidrains)](https://github.com/lucidrains). The optimizer is based on [this LAMB implementation](https://github.com/cybertronai/pytorch-lamb). See the corresponding licenses below.

<img src="https://peract.github.io/media/figures/sim_task.jpg" alt="drawing"/>

## Licenses
- [ARM License](https://github.com/stepjam/ARM/blob/main/LICENSE)
- [YARR Licence (Apache 2.0)](https://github.com/stepjam/YARR/blob/main/LICENSE)
- [RLBench Licence](https://github.com/stepjam/RLBench/blob/master/LICENSE)
- [PyRep License (MIT)](https://github.com/stepjam/PyRep/blob/master/LICENSE)
- [Perceiver PyTorch License (MIT)](https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE)
- [LAMB License (MIT)](https://github.com/cybertronai/pytorch-lamb/blob/master/LICENSE)
