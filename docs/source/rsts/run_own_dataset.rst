===============
Run own dataset
===============

.. _run_own_dataset:

    Run your own dataset with DF-VO is not complicated. 
    Basically, you need to add a dataset loader and update the configuration file.
    Here are the steps to run your own dataset.

    - Add dataset loader
    
        Refer to the example ``libs/datasets/adelaide.py``, there are some least functions you need to provide for your dataset loader.
        Some functions are optional where you would find "NotImplemented".
        Basically you need to have a fuction for loading camera intrinsics and image data.
        There are intrustions in the `libs/datasets/adelaide.py` as well.

    - Add the loader to Dataset

        After creating the dataset loader, add the dataset loader to ``libs/dataset/__init__.py``.
        You need to import the loader and put it in the dictionary `datasets` in the same file.
    
    - Update configuration file

        Update at least the following configurations in the config file. 

            - dataset
            - img_seq_dir
        

