.. role:: raw-html-m2r(raw)
   :format: html


Introduction
============

This repo implements the system described in the paper:

`Visual Odometry Revisited: What Should Be Learnt?  <https://arxiv.org/abs/1909.09803>`_ 

Huangying Zhan, Chamara Saroj Weerasekera, Jiawang Bian, Ian Reid

The demo video can be found `here <https://www.youtube.com/watch?v=Nl8mFU4SJKY>`_.

.. code-block::

   # The paper is accepted to ICRA-2020. Updated bibtex will be provided in the future.

   @article{zhan2019dfvo,
     title={Visual Odometry Revisited: What Should Be Learnt?},
     author={Zhan, Huangying and Weerasekera, Chamara Saroj and Bian, Jiawang and Reid, Ian},
     journal={arXiv preprint arXiv:1909.09803},
     year={2019}
   }

:raw-html-m2r:`<img src='../../misc/dfvo_eg.gif' width=640 height=320>`

This repo includes


#. the frame-to-frame tracking system **DF-VO**\ ;
#. evaluation scripts for visual odometry; 
#. trained models and VO results

Contents
^^^^^^^^


#. `Requirements <#part-1-requirements>`_
#. `Prepare dataset <#part-2-download-dataset-and-models>`_
#. `DF-VO <#part-3-DF-VO>`_
#. `Result evaluation <#part-4-result-evaluation>`_

Part 1. Requirements
^^^^^^^^^^^^^^^^^^^^

This code was tested with Python 3.6, CUDA 9.0, Ubuntu 16.04, and `PyTorch <https://pytorch.org/>`_.

We suggest use `Anaconda <https://www.anaconda.com/distribution/>`_ for installing the prerequisites.

.. code-block::

   conda env create -f requirement.yml -p dfvo # install prerequisites
   conda activate dfvo  # activate the environment [dfvo]

Part 2. Download dataset and models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main dataset used in this project is `KITTI Driving Dataset <http://www.cvlibs.net/datasets/kitti/eval_odometry.php>`_. After downloaing the dataset, create a softlink in the current repo.

.. code-block::

   ln -s KITTI_ODOMETRY/sequences dataset/kitti_odom/odom_data

For our trained models, please visit `here <https://www.dropbox.com/sh/9by21564eb0xloh/AABHFMlWd_ja14c5wU4R1KUua?dl=0>`_ to download the models and save the models into the directory ``model_zoo/``.

Part 3. DF-VO
^^^^^^^^^^^^^

The main algorithm is inplemented in ``vo_moduels.py``.
We have created different configurations for running the algrithm.

.. code-block::

   # Example 1: run default kitti setup
   python run.py -d options/kitti_default_configuration.yml  

   # Example 2: Run custom kitti setup
   # kitti_default_configuration.yml and kitti_stereo_0.yml are merged
   python run.py -c options/kitti/kitti_stereo_0.yml

The result (trajectory pose file) is saved in ``result_dir`` defined in the configuration file.
Please check the ``kitti_default_configuration.yml`` for more possible configuration.

Part 4. Result evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^

:raw-html-m2r:`<img src='../../misc/dfvo_result.png' width=320 height=480>`

Note that, we have cleaned and optimized the code for better readability and it changes the randomness such that the quantitative result is slightly different from the result reported in the paper. 

:raw-html-m2r:`<img src='../../misc/dfvo_result2.png' width=400 height=100>`

The original results, including related works, can be found `here <https://www.dropbox.com/sh/u7x3rt4lz6zx8br/AADshjd33Q3TLCy2stKt6qpJa?dl=0>`_.

KITTI
~~~~~

`KITTI Odometry benchmark <http://www.cvlibs.net/datasets/kitti/eval_odometry.php>`_ contains 22 stereo sequences, in which 11 sequences are provided with ground truth. The 11 sequences are used for evaluating visual odometry. 

.. code-block::

   python tool/evaluation/eval_odom.py --result result/tmp/0 --align 6dof

For more information about the evaluation toolkit, please check the `toolbox page <https://github.com/Huangying-Zhan/kitti_odom_eval>`_ or the `wiki page <https://github.com/Huangying-Zhan/DF-VO/wiki>`_.

Add your new dataset
^^^^^^^^^^^^^^^^^^^^


* configuration [seq, dataset, dataset_dir]
* dfvo.py datasets dictionary
* libs/datasets/\ **init**.py
* libs/datasets/DATASET_LOADER.py
* libs/deep_depth/monodepth2 (dataset parameters, min/max depth, stereo)
* libs/general/frame_drawer.py (vmax for depth)

License
^^^^^^^

For academic usage, the code is released under the permissive MIT license. Our intension of sharing the project is for research/personal purpose. For any commercial purpose, please contact the authors. 

Acknowledgement
^^^^^^^^^^^^^^^

Some of the codes were borrowed from the excellent works of `monodepth2 <https://github.com/nianticlabs/monodepth2>`_\ , `LiteFlowNet <https://github.com/twhui/LiteFlowNet>`_ and `pytorch-liteflownet <https://github.com/sniklaus/pytorch-liteflownet>`_. The borrowed files are licensed under their original license respectively.

To-do list
^^^^^^^^^^


* Release more pretrained models
* Release more results
* (maybe) training code: it takes longer time to clean the training code. Also, the current training code involves other projects which increases the difficulty in cleaning the code.
