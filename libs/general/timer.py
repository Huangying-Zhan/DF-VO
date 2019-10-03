# Copyright (C) Huangying Zhan 2019. All rights reserved.
#
# This software is licensed under the terms of the DF-VO licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np
from time import time


class Timers():
    def __init__(self, items=None):
        self.timers = {}
        if items is not None:
            self.add(items)

    def add(self, item):
        """add item to the timer
        Args:
            item (str/list): item name
        """
        if isinstance(item, list):
            for i in item:
                self.timers[i] = []
        elif isinstance(item, str):
            self.timers[item] = []
        else:
            assert False, "only list or str is accepted."
