'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 1970-01-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-11
@LastEditors: Huangying Zhan
@Description: 
'''

import os
import torch

from libs.general.utils import mkdir_if_not_exists

class CheckpointLogger():
    def __init__(self, epoch_freq, step_freq, result_dir):
        # # logging frequency
        # self.freqs = {}
        # self.freqs['epoch'] = epoch_freq
        # self.freqs['step'] = step_freq

        # directory
        self.result_dir = result_dir
    
    def save_checkpoint(self, item, cur_cnt, ckpt_data, is_best=False):
        """Save trained models, optimizer and training states
        
        Args:
            item (str): epoch / iter
            cur_counter (int): current counter
            ckpt_data (dict): checkpoint data dictionary
            
                - models: network model states
                - optimzier: optimizer state
                - train_state: extra information
                    - epoch
                    - step
            is_best (bool): model with best validation loss
        """
        models = ckpt_data['models']
        optimizer = ckpt_data['optimizer']
        train_state = ckpt_data['train_state']

        # Save current checkpoint
        save_folder = os.path.join(
                self.result_dir, "models", 
                "{}_{}".format(item, cur_cnt)
                )
        mkdir_if_not_exists(save_folder)

        print("==> Save checkpoint at {} {}".format(item, cur_cnt))
        self.save_model(save_folder, models)
        self.save_optimizer(save_folder, optimizer)
        self.save_train_state(save_folder, train_state)

        # Save best model
        if is_best:
            save_folder = os.path.join(
                self.result_dir, "models", "best"
                )
            mkdir_if_not_exists(save_folder)
            print("==> Save best model.")
            self.save_model(save_folder, models)
            self.save_optimizer(save_folder, optimizer)
            self.save_train_state(save_folder, train_state)
            with open(os.path.join(save_folder, "best.txt"), 'w') as f:
                line = "{}: {}".format(item, cur_cnt)
                f.writelines(line)
    
    def save_model(self, save_folder, models):
        """Save model checkpoints
        Args:
            save_folder (str): directory for saving models
            models (dict): model dictionary
        """
        for model_name, model in models.items():
            ckpt_path = os.path.join(save_folder, "{}.pth".format(model_name))
            torch.save(model.state_dict(), ckpt_path)
    
    def save_optimizer(self, save_folder, optimizer):
        """Save optimizer data
        Args:
            save_folder (str): directory for saving models
            optimizer (torch.optim): torch optimizer data
        """
        ckpt_path = os.path.join(save_folder, "optimizer.pth")
        torch.save(optimizer.state_dict(), ckpt_path)
    
    def save_train_state(self, save_folder, train_state):
        """Save optimizer data
        Args:
            save_folder (str): directory for saving models
            train_state (dict): extra training state information
        """
        ckpt_path = os.path.join(save_folder, "train_state.pth")
        torch.save(train_state, ckpt_path)