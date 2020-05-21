''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-21
@LastEditors: Huangying Zhan
@Description: Timer object for counting times.
'''

import numpy as np
from time import time


class Timer():
    """Timer class to count time and do time analysis
    """

    def __init__(self, items=None):
        """
        Args:
            items (list/str): list of items to be counted, each item is a str
        """
        self.timers = {}
        if items is not None:
            self.add(items)

    def add(self, item, group=None):
        """add item to the timer
        
        Args:
            item (str/list): item name
            group (str): group name of the item
        """
        if isinstance(item, list):
            for i in item:
                self.timers[i] = {
                    'name': i,
                    'time': 0,
                    'is_counting': False,
                    'duration': [],
                    'group': group
                }
        elif isinstance(item, str):
            self.timers[item] = {
                    'name': item,
                    'time': 0,
                    'is_counting': False,
                    'duration': [],
                    'group': group
                }
        else:
            assert False, "only list or str is accepted."
    
    def start(self, item, group=None):
        """Start timer for an item

        Args:
            item (str): timer name
            group (str): group name for the item
        """
        if self.timers.get(item, -1) == -1:
            self.add(item, group)

        assert not(self.timers[item]['is_counting']),  "Timer for {} has started already.".format(item)
        
        self.timers[item]['is_counting'] = True
        self.timers[item]['time'] = time()
    
    def end(self, item):
        """Stop timer for an item

        Args:
            item (str): timer name
        """
        assert self.timers[item]['is_counting'], "Timer for {} has not started.".format(item)
        
        duration = time() - self.timers[item]['time']
        self.timers[item]['duration'].append(duration)
        self.timers[item]['is_counting'] = False
    
    def time_analysis(self):
        """Time analysis of the items
        """
        print("----- time breakdown -----")
        # group items according to groups
        group_timers = {'single': []}
        for key in sorted(self.timers.keys()):
            group_name = self.timers[key]['group']
            if group_name is not None:
                if group_timers.get(group_name, -1) == -1:
                    group_timers[group_name] = []
                group_timers[group_name].append(self.timers[key])
            else:
                group_timers['single'].append(self.timers[key])
        
        # display times
        for group_name, members in group_timers.items():
            print("Group [{}]: ".format(group_name))
            group_avg_times = []
            for member in members:
                avg_time = np.asarray(member['duration']).mean()
                group_avg_times.append(avg_time)
                print("\t[{}]: {:.03f}s".format(member['name'], avg_time))
