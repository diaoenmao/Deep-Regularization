#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python pipe.py --seed 0 --resume_mode 0 --control_name MNIST_linear_pqiproximal_5.0_SGD_inplace_0.5_False_1.0_2.0&
CUDA_VISIBLE_DEVICES="1" python pipe.py --seed 0 --resume_mode 0 --control_name MNIST_linear_pqiproximal_5.0_SGD_inplace_1.0_False_1.0_2.0&
CUDA_VISIBLE_DEVICES="2" python pipe.py --seed 0 --resume_mode 0 --control_name MNIST_linear_pqiproximal_5.0_SGD_inplace_2.0_False_1.0_2.0&
CUDA_VISIBLE_DEVICES="3" python pipe.py --seed 0 --resume_mode 0 --control_name MNIST_linear_pqiproximal_10.0_SGD_inplace_0.5_False_1.0_2.0&
CUDA_VISIBLE_DEVICES="0" python pipe.py --seed 0 --resume_mode 0 --control_name MNIST_linear_pqiproximal_10.0_SGD_inplace_1.0_False_1.0_2.0&
CUDA_VISIBLE_DEVICES="1" python pipe.py --seed 0 --resume_mode 0 --control_name MNIST_linear_pqiproximal_10.0_SGD_inplace_2.0_False_1.0_2.0&
CUDA_VISIBLE_DEVICES="2" python pipe.py --seed 0 --resume_mode 0 --control_name CIFAR10_linear_pqiproximal_5.0_SGD_inplace_0.5_False_1.0_2.0&
CUDA_VISIBLE_DEVICES="3" python pipe.py --seed 0 --resume_mode 0 --control_name CIFAR10_linear_pqiproximal_5.0_SGD_inplace_1.0_False_1.0_2.0
wait
CUDA_VISIBLE_DEVICES="0" python pipe.py --seed 0 --resume_mode 0 --control_name CIFAR10_linear_pqiproximal_5.0_SGD_inplace_2.0_False_1.0_2.0&
CUDA_VISIBLE_DEVICES="1" python pipe.py --seed 0 --resume_mode 0 --control_name CIFAR10_linear_pqiproximal_10.0_SGD_inplace_0.5_False_1.0_2.0&
CUDA_VISIBLE_DEVICES="2" python pipe.py --seed 0 --resume_mode 0 --control_name CIFAR10_linear_pqiproximal_10.0_SGD_inplace_1.0_False_1.0_2.0&
CUDA_VISIBLE_DEVICES="3" python pipe.py --seed 0 --resume_mode 0 --control_name CIFAR10_linear_pqiproximal_10.0_SGD_inplace_2.0_False_1.0_2.0&
wait