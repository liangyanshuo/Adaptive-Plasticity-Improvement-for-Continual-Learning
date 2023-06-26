# Adaptive Plasticity Improvement for Continual Learning

Code for CVPR 2023 paper [Adaptive Plasticity Improvement for Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Liang_Adaptive_Plasticity_Improvement_for_Continual_Learning_CVPR_2023_paper.pdf).

## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.8.9
- torch = 1.10.0
- torchvision = 0.11.1

## run for experiments

```
bash script/perform.sh Your_data_path Your_CUDA_id
```

## Citation

If you find this code useful, please consider citing:
```bibtex
@inproceedings{liang2023adaptive,
  title={Adaptive Plasticity Improvement for Continual Learning},
  author={Liang, Yan-Shuo and Li, Wu-Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7816--7825},
  year={2023}
}
```

We would like to thank the authors of the following repositories (from which we borrowed code). </br>
* https://github.com/danruod/FS-DGPM


