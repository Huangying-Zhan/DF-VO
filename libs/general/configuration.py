'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-10-13
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-17
@LastEditors: Huangying Zhan
@Description: Configuration Class to update/merge/write configurations
'''

from easydict import EasyDict as edict

from libs.general.utils import read_yaml


class ConfigLoader():    
    def update_dict(self, dict1, dict2):
        """Update dict1 according to dict2

        Args:
            dict1 (dict): reference dictionary
            dict2 (dict): new dictionary
        
        Returns:
            dict1 (dict): updated reference dictionary
        """
        for item in dict2:
            if dict1.get(item, -1) != -1:
                if isinstance(dict1[item], dict):
                    dict1[item] = self.update_dict(dict1[item], dict2[item])
                else:
                    dict1[item] = dict2[item]
            else:
                dict1[item] = dict2[item]
        return dict1


    def merge_cfg(self, cfg_files):
        """Merge default configuration and custom configuration

        Args:
            cfg_files (str): configuration file paths [default, custom]

        Returns:
            cfg (edict): merged EasyDict
        """
        cfg = {}
        for f in cfg_files:
            if f is not None:
                cfg = self.update_dict(cfg, read_yaml(f))
        return edict(cfg)

    def write_cfg(self, default, merge, f, level_cnt=0):
        """Write configuration to file
        
        Args:
            default (dict): default configuration dictionary
            merge (dict): merged configuration dictionary
            file (TextIOWrapper): text IO wrapper object
        """
        offset_len = 100
        for item in merge:
            if isinstance(merge[item], dict):
                # go deeper for dict item
                line = "  "*level_cnt + item + ": "
                offset = offset_len - len(line)
                line += " "*offset + " # | "
                
                # check if default has this config
                if default.get(item, -1) == -1:
                    default[item] = {}
                    line += " --NEW-- "
                f.writelines(line + "\n")
                self.write_cfg(default[item], merge[item], f, level_cnt+1)
            else:
                # write current config
                line = "  " * level_cnt + item + ": "
                if merge[item] is not None:
                    line += str(merge[item])
                else:
                    line += " "
                
                offset = offset_len - len(line)
                line += " "*offset + " # | "
                f.writelines(line)

                # write default if default is different from current
                if default.get(item, -1) != -1:
                    line = " "
                    if merge[item] != default[item]:
                        line = str(default[item])
                    f.writelines(line)
                else:
                    line = " --NEW-- "
                    f.writelines(line)
                f.writelines("\n")

    def save_cfg(self, cfg_files, file_path):
        """Save configuration file

        Args:
            cfg_files (str): configuration file paths [default, custom]
            file_path (str): path of text file for writing the configurations
        """
        # read configurations
        default = read_yaml(cfg_files[0])
        merged = self.merge_cfg(cfg_files)

        # create file to be written
        f = open(file_path, 'w')

        # write header line
        line = "# " + "-"*20 + " Setup " + "-"*74
        line += "|" + "-"*10 + " Default " + "-"*20 + "\n"
        f.writelines(line)

        # write configurations
        self.write_cfg(default, merged, f)
        f.close()
