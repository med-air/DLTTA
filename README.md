# DLTTA: Dynamic Learning Rate for Test-time Adaptation on Cross-domain Medical Images
Pytorch implementation for TMI paper DLTTA: Dynamic Learning Rate for Test-time Adaptation on Cross-domain Medical Images, by [Hongzheng Yang](https://github.com/HongZhengYang), [Chen Cheng](https://cchen-cc.github.io/), [Meirui Jiang](https://meiruijiang.github.io/MeiruiJiang/), [Quande Liu](https://liuquande.github.io/),  [Jianfeng Cao](), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/).  

## Abstract



![](assets/Figure.png)

## Files

In this repository, we provide the implementation of our dynamic learning rate method on OCT dataset. The ATTA and Tent implementation were adopted from their official implementation. ([Tent](https://github.com/DequanWang/tent), [ATTA]())

To reproduce results on Camelyon17 and Prostate datasets, please refer to the experiments folder.

## Datasets

The OCT dataset can downloaded from [here](http://iacl.ece.jhu.edu/index.php?title=Resources). 

The Camelyon17 dataset can be downloaded from [here](https://wilds.stanford.edu/).

The Prostate dataset can be downloaded from [here](https://liuquande.github.io/SAML/).

## Usage

1. create conda environment
   
       conda create -n DLTTA python=3.7
       conda activate DLTTA
   
2. Install dependencies:

   1. install pytorch==1.7.0 torchvision==0.9.0 (via conda, recommend)

3. download the dataset

4. download the pretrained model from [google drive](https://drive.google.com/drive/folders/1RyTLkoLf1w_7rgvQl_aEERd-8YKGc0Yt?usp=sharing)

5. modify the corresponding data path and model path in test.sh

6. run test.sh to adapt the model

## Citation

If this repository is useful for your research, please cite:

    @article{2022DLTTA,
     title={DLTTA: Dynamic Learning Rate for Test-time Adaptation on Cross-domain Medical Images},
      author={Hongzheng Yang, Cheng Chen, Meirui Jiang, Quande Liu, Jianfeng Cao, Pheng Ann Heng, Qi Dou},
      year={2022}
    }  

### Questions

Please feel free to contact 'hzyang05@gmail.com' if you have any questions. 

