Flow-GAN: Combining Maximum Likelihood and Adversarial Learning in Generative Models
============================================

This repository provides a reference implementation for learning Flow-GAN models as described in the paper:


> Flow-GAN: Combining Maximum Likelihood and Adversarial Learning in Generative Models  
[Aditya Grover](https://aditya-grover.github.io), [Manik Dhar](https://web.stanford.edu/~dmanik/), and [Stefano Ermon](https://cs.stanford.edu/~ermon/).  
AAAI Conference on Artificial Intelligence (AAAI), 2018.   
Paper: https://arxiv.org/pdf/1705.08868.pdf  
Blog post: https://ermongroup.github.io/blog/flow-gan

## Requirements

The codebase is implemented in Python 3.6. To install the necessary requirements, run the following commands:

```
pip install -r requirements.txt
```

## Datasets

The scripts for downloading and loading the MNIST and CIFAR10 datasets are included in the `datasets_loader` folder. These scripts will be called automatically the first time the `main.py` script is run.

## Options

Learning and inference of Flow-GAN models is handled by the `main.py` script which provides the following command line arguments.

```
  --beta1 FLOAT           beta1 parameter for Adam optimizer
  --epoch INT             number of epochs to train
  --batch_size FLOAT      training batch size
  --learning_rate FLOAT   learning rate
  --input_height INT      The size of image to use
  --input_width INT       The size of image to use if none given use same value as input height
  --c_dim INT             Dimension of image color
  --dataset STR           The name of dataset [mnist, svhn, cifar-10]
  --checkpoint_dir STR    Directory name to save the checkpoints
  --log_dir STR           Directory name to save the logs
  --sample_dir STR        Directory name to save the image samples
  --f_div STR             divergence used for specifying the gan objective
  --prior STR             prior for generator
  --alpha FLOAT           alpha value for applying logits
  --lr_decay FLOAT        Learning rate decay rate
  --min_lr FLOAT          minimum lr allowed on decay
  --reg FLOAT             regularization parameter for adversarial training
  --model_type STR        real_nvp or nice
  --n_critic INT          no of discriminator iterations
  --no_of_layers INT      No of units between input and output in the m function for a coupling layer
  --hidden_layers INT     Size of hidden layers (applicable only for NICE)
  --like_reg FLOAT        regularizing factor for likelihood vs. adversarial losses for hybrid
  --df_dim FLOAT          Dim depth for discriminator
```


## Examples

### Training flow-GAN models on the MNIST dataset with NICE architecture.

*Maximum Likelihood Estimation (MLE)*
```
python main.py --dataset mnist --input_height=28 --c_dim=1  --checkpoint_dir checkpoint_mnist/mle --sample_dir samples_mnist/mle --model_type nice --log_dir logs_mnist/mle 
--prior logistic --beta1 0.5 --learning_rate 1e-4 --alpha 1e-7 --epoch 500 --batch_size 100 --like_reg 1.0 --n_critic 0 --no_of_layers 5
```

*Adversarial training (ADV)*
```
python main.py --dataset mnist --input_height=28 --c_dim=1  --checkpoint_dir checkpoint_mnist/gan --sample_dir samples_mnist/gan --model_type nice --log_dir logs_mnist/gan 
--prior logistic --beta1 0.5 --learning_rate 1e-4 --alpha 1e-7 --reg 10.0 --epoch 500 --batch_size 100 --like_reg 0.0 --n_critic 5 --no_of_layers 5
```

*Hybrid* 
```
python main.py --dataset mnist --input_height=28 --c_dim=1  --checkpoint_dir checkpoint_mnist/flow --sample_dir samples_mnist/flow --model_type nice --log_dir logs_mnist/flow 
--prior logistic --beta1 0.5 --learning_rate 1e-4 --alpha 1e-7 --reg 10.0 --epoch 500 --batch_size 100 --like_reg 1.0 --n_critic 5 --no_of_layers 5
```

### Training flow-GAN models on the CIFAR dataset with Real-NVP architecture.

*Maximum Likelihood Estimation (MLE)*
```
python main.py --dataset cifar --input_height=32 --c_dim=3  --checkpoint_dir checkpoint_cifar/mle --sample_dir samples_cifar/mle --model_type real_nvp --log_dir logs_cifar/mle 
--prior gaussian --beta1 0.9 --learning_rate 1e-3 --alpha 1e-7 --epoch 300 --lr_decay 0.999995 --batch_size 64 --like_reg 1.0 --n_critic 0 --no_of_layers 8 --batch_norm_adaptive 0
```


*Adversarial training (ADV)*
```
python main.py --dataset cifar --input_height=32 --c_dim=3  --checkpoint_dir checkpoint_cifar/gan --sample_dir samples_cifar/gan --model_type real_nvp --log_dir logs_cifar/gan 
--prior gaussian --beta1 0.5 --learning_rate 1e-4 --alpha 1e-7 --epoch 300 --batch_size 64 --like_reg 0.0  --n_critic 5 --no_of_layers 8
```


*Hybrid*
```
python main.py --dataset cifar --input_height=32 --c_dim=3  --checkpoint_dir checkpoint_cifar/flow --sample_dir samples_cifar/flow --model_type real_nvp --log_dir logs_cifar/flow 
--prior gaussian --beta 0.5 --learning_rate 1e-3 --lr_decay 0.99999 --alpha 1e-7 --epoch 500 --batch_size 64 --like_reg 20.  --n_critic 5 --no_of_layers 8
```

Portions of the codebase in this repository uses code originally provided in the open-source [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow) and [Real-NVP](https://github.com/taesung89/real-nvp) repositories. 

## Citing

If you find flow-GANs useful in your research, please consider citing the following paper:


>@inproceedings{grover2018flowgan,  
  title={Flow-GAN: Combining Maximum Likelihood and Adversarial Learning in Generative Models},  
  author={Grover, Aditya and Dhar, Manik and Ermon, Stefano},  
  booktitle={AAAI Conference on Artificial Intelligence},  
  year={2018}}