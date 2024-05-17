# FCS: Feature Calibration and Separation for Non-Exemplar Class Incremental Learning

Official implementation of "[FCS: Feature Calibration and Separation for Non-Exemplar Class Incremental Learning](https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers)"


<p align="center"><img src="./files/pipeline-fcs.png" align="center" width="750"></p>


## Requirements

### Environment
Python 3.7.13

PyTorch 1.8.1



## Run commands
Json files for different experinments are provided in ./exps/fcs/

### Run algorithms on CIFAR100-5stages
```shell
python main.py --config=./exps/fcs/cifar100/5/first_stage.json # base stage
python main.py --config=./exps/fcs/cifar100/5/second_stage.json # incremental learning
```
## Results

Results for different experinments are provided in ./files/results.txt
## Acknowledgement

This project is mainly based on [PyCIL](https://github.com/G-U-N/PyCIL).

## Citation

If you find this work helpful, please cite:
```
@article{,
  title={FCS: Feature Calibration and Separation for Non-Exemplar Class Incremental Learning},
  author={Qiwei Li, Yuxin Peng, Jiahuan Zhou},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```