import numpy as np
from dataset import dataset_train, dataset_test
from nmf import train, test
import yaml
import os

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset_dir = config['dataset_dir']
model_dir = config['model_dir']
num_iters_train = config['num_iters_train']
num_iters_test = config['num_iters_test']
num_bases = config['num_bases']

def main():
    # train
    train_instr_feat, train_vocal_feat = dataset_train()
    B_vocal, B_instr = train(train_vocal_feat, train_instr_feat, num_bases=num_bases, num_iters=num_iters_train)
    del train_instr_feat, train_vocal_feat
    np.save(os.path.join(model_dir, 'B_vocal.npy'), B_vocal)
    np.save(os.path.join(model_dir, 'B_instr.npy'), B_instr)

    # test
    test_feats, test_phases, test_files = dataset_test()
    test(test_feats, test_phases, test_files, B_vocal, B_instr, num_iters=num_iters_test)

if __name__ == '__main__':
    main()