#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python pipe.py --data_name MNIST --model_name linear --init_seed 0 --num_experiments 5 --regularizer_name none&
CUDA_VISIBLE_DEVICES="1" python pipe.py --data_name MNIST --model_name linear --init_seed 0 --num_experiments 5 --regularizer_name l1_softthreshold --lambda 0.01&
CUDA_VISIBLE_DEVICES="2" python pipe.py --data_name MNIST --model_name linear --init_seed 0 --num_experiments 5 --regularizer_name l1_proximal --lambda 0.01&
CUDA_VISIBLE_DEVICES="3" python pipe.py --data_name MNIST --model_name linear --init_seed 0 --num_experiments 5 --regularizer_name l2 --lambda 0.01&
CUDA_VISIBLE_DEVICES="0" python pipe.py --data_name MNIST --model_name linear --init_seed 0 --num_experiments 5 --regularizer_name pqi_proximal --lambda 0.1&
CUDA_VISIBLE_DEVICES="1" python pipe.py --data_name MNIST --model_name linear --init_seed 0 --num_experiments 5 --regularizer_name pqi_proximal --lambda 1.0&
CUDA_VISIBLE_DEVICES="2" python pipe.py --data_name MNIST --model_name linear --init_seed 0 --num_experiments 5 --regularizer_name l1_sgd_naive --lambda 0.01
wait