from pickletools import optimize
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.data_loaders import get_eval_loader

import torchvision.transforms as transforms
from examples.transforms import initialize_transform
from examples.configs.utils import populate_defaults
import examples.configs.supported as supported
from examples.utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool, get_model_prefix, move_to

from configs.supported import algo_log_metrics, losses
from losses import initialize_loss

import torch
import argparse
import wilds
import logging
from networks.models import DenseNet121
from ttt import rotate_batch
from  tqdm import tqdm
import sys
import numpy as np
import random
import torch.backends.cudnn as cudnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, default='camelyon17')
    parser.add_argument('--algorithm',  choices=supported.algorithms, default='ERM')
    parser.add_argument('--root_dir', default='/research/pheng4/qdliu/hzyang/wilds/examples/')

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to download the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

    # Unlabeled Dataset
    parser.add_argument('--unlabeled_split', default=None, type=str, choices=wilds.unlabeled_splits,  help='Unlabeled split to use. Some datasets only have some splits available.')
    parser.add_argument('--unlabeled_version', default=None, type=str, help='WILDS unlabeled dataset version number.')
    parser.add_argument('--use_unlabeled_y', default=False, type=parse_bool, const=True, nargs='?', 
                        help='If true, unlabeled loaders will also the true labels for the unlabeled data. This is only available for some datasets. Used for "fully-labeled ERM experiments" in the paper. Correct functionality relies on CrossEntropyLoss using ignore_index=-100.')

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--unlabeled_loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--unlabeled_n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--unlabeled_batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

    # Model
    parser.add_argument('--model', choices=supported.models)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument('--noisystudent_add_dropout', type=parse_bool, const=True, nargs='?', help='If true, adds a dropout layer to the student model of NoisyStudent.')
    parser.add_argument('--noisystudent_dropout_rate', type=float)
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')
    parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')

    # NoisyStudent-specific loading
    parser.add_argument('--teacher_model_path', type=str, help='Path to NoisyStudent teacher model weights. If this is defined, pseudolabels will first be computed for unlabeled data before anything else runs.')

    # Transforms
    parser.add_argument('--transform', choices=supported.transforms)
    parser.add_argument('--additional_train_transform', choices=supported.additional_transforms, help='Optional data augmentations to layer on top of the default transforms.')
    parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--resize_scale', type=float)
    parser.add_argument('--max_token_length', type=int)
    parser.add_argument('--randaugment_n', type=int, help='Number of RandAugment transformations to apply.')

    # Objective
    parser.add_argument('--loss_function', choices=supported.losses)
    parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--dann_penalty_weight', type=float)
    parser.add_argument('--dann_classifier_lr', type=float)
    parser.add_argument('--dann_featurizer_lr', type=float)
    parser.add_argument('--dann_discriminator_lr', type=float)
    parser.add_argument('--afn_penalty_weight', type=float)
    parser.add_argument('--safn_delta_r', type=float)
    parser.add_argument('--hafn_r', type=float)
    parser.add_argument('--use_hafn', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--self_training_lambda', type=float)
    parser.add_argument('--self_training_threshold', type=float)
    parser.add_argument('--pseudolabel_T2', type=float, help='Percentage of total iterations at which to end linear scheduling and hold lambda at the max value')
    parser.add_argument('--soft_pseudolabels', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--algo_log_metric')
    parser.add_argument('--process_pseudolabels_function', choices=supported.process_pseudolabels_functions)

    # Model selection
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Misc
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

    # Weights & Biases
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    config = parser.parse_args()
    config = populate_defaults(config)
    logging.basicConfig(filename="log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(config))

    seed = 0
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    full_dataset = get_dataset(dataset="camelyon17", root_dir = '/research/pheng4/qdliu/hzyang/test_time_medical/dataset/')
     
    train_transform = initialize_transform(
            transform_name=config.transform,
            config=config,
            dataset=full_dataset,
            additional_transform_name=config.additional_train_transform,
            is_training=True)
    eval_transform = initialize_transform(
            transform_name=config.transform,
            config=config,
            dataset=full_dataset,
            is_training=False)

    # Get the training set

    train_data = full_dataset.get_subset(
        "train",
        transform=train_transform,
    )

    val_data = full_dataset.get_subset(
        "test",
        transform=eval_transform,
    )


    train_loader = get_train_loader("standard", train_data, batch_size=32)

    val_loader = get_eval_loader("standard", val_data, batch_size=32)

    model = DenseNet121(out_size=2, drop_rate=0.2)
    model = model.cuda()
    loss = initialize_loss(config.loss_function, config)
    metric = algo_log_metrics[config.algo_log_metric]
    epoches = 10
    # Train loop
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    epoch = 0
    torch.save(model.state_dict(), f'checkpoint/epoch_{epoch}.pth')
    for epoch in  range(epoches):
        total  = 0
        correct  = 0 
        model.train()   
        for labeled_batch in tqdm(train_loader):
            x, y, metadata = labeled_batch
            
            x_rotate, y_rotate = rotate_batch(x, 'rand')
            x_rotate, y_rotate, y = x_rotate.cuda(), y_rotate.cuda(), y.cuda()
            _, output, ssh_output = model(x_rotate)
            loss = criterion(output, y) + criterion(ssh_output, y_rotate)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += y.size(0)
            pred = output.data.max(1)[1]
            batch_correct = pred.eq(y.view(-1)).sum().item()
            correct += batch_correct
        acc = correct / total
        logging.info('Train Epoch: ' + str(epoch))
        logging.info('Accuracy: ' + str(acc))
        torch.save(model.state_dict(), f'checkpoint/epoch_{epoch}.pth')
        model.eval()
        correct = 0
        total = 0
        for labeled_batch in tqdm(val_loader):
            x, y, metadata = labeled_batch
            x, y = x.cuda(), y.cuda()
            _, output, ssh_output = model(x)
            
            total += y.size(0)
            pred = output.data.max(1)[1]
            batch_correct = pred.eq(y.view(-1)).sum().item()
            correct += batch_correct
        acc = correct / total
        logging.info('Test Epoch: ' + str(epoch))
        logging.info('Accuracy: ' + str(acc))
        
        
    
