import argparse
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--mode', default="base", type=str)         # always should be base
parser.add_argument('--run', default='train', type=str)         # train or test
parser.add_argument('--world_size', default=1, type=int)        # distributed training, deprecated 

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



def make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + world_size + num_experiments + resume_mode + control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    run = args['run']
    init_gpu = args['init_gpu']
    num_gpus = args['num_gpus']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    mode = args['mode']
    split_round = args['split_round']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in
               list(range(init_gpu, init_gpu + num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    filename = f'{run}_{mode}'
    if mode == 'base':
        script_name = [[f'{run}_model.py']]
        data_name = ['MNIST', 'CIFAR10']
        model_name = ['linear', 'mlp', 'cnn', 'resnet18']
        control_name = [[data_name, model_name]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    else:
        raise ValueError('Not valid mode')
    s = '#!/bin/bash\n'
    j = 1
    filename_idx = 1
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python src/{} --init_seed {} --world_size {} --num_experiments {} --resume_mode {} --control_name {}&\n'.format(gpu_ids[i % len(gpu_ids)], *controls[i])
        if i % round == round - 1:
            s = s[:-2] + '\nwait\n'
            if j % split_round == 0:
                print(s)
                run_file = open(f"{filename}_{filename_idx}.sh", 'w')
                run_file.write(s)
                run_file.close()
                s = '#!/bin/bash\n'
                filename_idx += 1
            j = j + 1
    if s != '#!/bin/bash\n':
        if s[-5:-1] != 'wait':
            s = s + 'wait\n'
        print(s)
        run_file = open(f'{filename}_{filename_idx}.sh', 'w')
        run_file.write(s)
        run_file.close()
    return


if __name__ == '__main__':
    main()
