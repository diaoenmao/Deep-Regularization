#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python src/test_model.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_linear&
CUDA_VISIBLE_DEVICES="1" python src/test_model.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_mlp&
CUDA_VISIBLE_DEVICES="2" python src/test_model.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_cnn&
CUDA_VISIBLE_DEVICES="3" python src/test_model.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_resnet18&
CUDA_VISIBLE_DEVICES="0" python src/test_model.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_linear&
CUDA_VISIBLE_DEVICES="1" python src/test_model.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_mlp&
CUDA_VISIBLE_DEVICES="2" python src/test_model.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_cnn&
CUDA_VISIBLE_DEVICES="3" python src/test_model.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_resnet18
wait
