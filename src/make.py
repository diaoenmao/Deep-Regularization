import argparse
import itertools
from pprint import pprint

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)         # train or test

# will use GPUs from range(init_gpu, init_gpu + num_gpus) 
parser.add_argument('--init_gpu', default=0, type=int)          # which GPU to start 
parser.add_argument('--num_gpus', default=4, type=int)          # how many GPUs to use

# will run experiments with seeds range(init_seed, init_seed + num_experiments, experiment_step)
parser.add_argument('--init_seed', default=0, type=int)         # random seed for experiement 
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--experiment_step', default=1, type=int)

# after every 'round' rounds, the bash script will wait for a moment 
parser.add_argument('--round', default=4, type=int)             

# after every 'split_round' rounds, we will generate a new bash script 
parser.add_argument('--split_round', default=65535, type=int)

# resume mode: 0 means start over and 1 means start at checkpoint 
parser.add_argument('--resume_mode', default=0, type=int)

args = vars(parser.parse_args())


def make_controls(seeds, resume_mode, control_name):
    if "none" in control_name[0][2]: 
        controls = []
    
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
        
    control_names = [control_names]
    controls = seeds + resume_mode + control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    # set up GPUS 
    init_gpu = args['init_gpu']
    num_gpus = args['num_gpus']
    gpu_ids = list(range(init_gpu, init_gpu + num_gpus))
    
    
    script_name = "pipe.py" 
    
    init_seed = args["init_seed"]
    num_experiments = args["num_experiments"]
    experiment_step = args["experiment_step"]
        
    
    round = args['round']
    split_round = args['split_round']
    
    resume_mode = args['resume_mode']
    
    run = args['run']
    
    
    seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    
    
    data_name = ['MNIST', 'CIFAR10']
    model_name = ['linear']
    regularizer_name = ["none", "l1softthreshold", "l1proximal", "l2", "pqiproximal"]
    lmbda = ["1.0", "0.1", "0.01"]
    reg_optimizer = ["SGD"]
    reg_initialization = ["inplace"]
    clipping_scale = ["1.0"]
    line_crossing = ["False"]
    p = ["1.0"] 
    q = ["2.0"]
    control_name = [
        # [data_name, model_name, ["none"], ["0.0"], ["SGD"], ["inplace"], ["1.0"], ["False"], ["1.0"], ["2.0"]], 
        # [data_name, model_name, ["l1softthreshold"], lmbda, ["SGD"], ["inplace"], ["1.0"], ["False"], ["1.0"], ["2.0"]], 
        # [data_name, model_name, ["l1proximal"], lmbda, ["SGD"], ["zeros", "rand", "inplace"], ["0.5", "1.0", "2.0"], ["False"], ["1.0"], ["2.0"]], 
        # [data_name, model_name, ["l2"], lmbda, ["SGD"], ["inplace"], ["1.0"], ["False"], ["1.0"], ["2.0"]], 
        [data_name, model_name, ["pqiproximal"], ["5.0", "10.0"], ["SGD"], ["inplace"], ["0.5", "1.0", "2.0"], ["False"], ["1.0"], ["2.0"]]
    ]
    controls = make_controls(seeds, resume_mode, control_name)
    print(f"Commands to Run : {len(controls)}")
    
    s = '#!/bin/bash\n'
    j = 1
    script_idx = 1
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python pipe.py --seed {} --resume_mode {} --control_name {}&\n'.format(gpu_ids[i % len(gpu_ids)], *controls[i])
        if i % round == round - 1:
            s = s[:-2] + '\nwait\n'
            if j % split_round == 0:
                # print(s)
                run_file = open(f"script_{script_idx}.sh", 'w')
                run_file.write(s)
                run_file.close()
                s = '#!/bin/bash\n'
                script_idx += 1
            j = j + 1
    if s != '#!/bin/bash\n':
        if s[-5:-1] != 'wait':
            s = s + 'wait\n'
        # print(s)
        run_file = open(f'script_{script_idx}.sh', 'w')
        run_file.write(s)
        run_file.close()
        
    print(f"Bash script file successfully created. ")
    return


if __name__ == '__main__':
    main()
