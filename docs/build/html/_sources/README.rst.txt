.. role:: raw-html-m2r(raw)
   :format: html


Introduction
============

This repo implements the system described in the ICRA-2020 paper and the extended report:

`Visual Odometry Revisited: What Should Be Learnt?  <https://arxiv.org/abs/1909.09803>`_ 

`DF-VO: What Should Be Learnt for Visual Odometry? <https://arxiv.org/abs/2103.00933>`_ 

Huangying Zhan, Chamara Saroj Weerasekera, Jiawang Bian, Ravi Garg, Ian Reid

The demo video can be found `here <https://www.youtube.com/watch?v=Nl8mFU4SJKY>`_.

.. code-block::

   @INPROCEEDINGS{zhan2019dfvo,
     author={H. {Zhan} and C. S. {Weerasekera} and J. -W. {Bian} and I. {Reid}},
     booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)}, 
     title={Visual Odometry Revisited: What Should Be Learnt?}, 
     year={2020},
     volume={},
     number={},
     pages={4203-4210},
     doi={10.1109/ICRA40945.2020.9197374}}

   @misc{zhan2021dfvo,
         title={DF-VO: What Should Be Learnt for Visual Odometry?}, 
         author={Huangying Zhan and Chamara Saroj Weerasekera and Jia-Wang Bian and Ravi Garg and Ian Reid},
         year={2021},
         eprint={2103.00933},
         archivePrefix={arXiv},
         primaryClass={cs.CV}
   }

:raw-html-m2r:`<img src='https://raw.githubusercontent.com/Huangying-Zhan/DF-VO/docs/docs/source/misc/dfvo_eg.gif' width=640 height=320>`

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

This code was tested with Python 3.6, CUDA 9.0, Ubuntu 16.04, and `PyTorch-1.1 <https://pytorch.org/>`_.

We suggest use `Anaconda <https://www.anaconda.com/distribution/>`_ for installing the prerequisites.

.. code-block::

   cd envs
   conda env create -f requirement.yml -p {ANACONDA_DIR/envs/dfvo} # install prerequisites
   conda activate dfvo  # activate the environment [dfvo]

Part 2. Download dataset and models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main dataset used in this project is `KITTI Driving Dataset <http://www.cvlibs.net/datasets/kitti/eval_odometry.php>`_. After downloaing the dataset, create a softlink in the current repo.

.. code-block::

   ln -s KITTI_ODOMETRY/sequences dataset/kitti_odom/odom_data

For our trained models, please visit `here <https://www.dropbox.com/sh/9by21564eb0xloh/AABHFMlWd_ja14c5wU4R1KUua?dl=0>`_ to download the models and save the models into the directory ``model_zoo/``.

Part 3. DF-VO
^^^^^^^^^^^^^

We have created some examples for running the algorithm.

.. code-block::

   # Example 1: run default kitti setup
   python apis/run.py -d options/examples/default_configuration.yml  

   # Example 2: Run custom kitti setup
   python apis/run.py \
   -d options/examples/default_configuration.yml \
   -c options/examples/kitti_stereo_train_icra.yml \
   --no_confirm

   # More examples and our experiments can be found in scripts/experiment.sh

The result (trajectory pose file) is saved in ``result_dir`` defined in the configuration file.
Please check the ``options/examples/default_configuration.yml`` or `Configuration Documentation <https://df-vo.readthedocs.io/en/latest/rsts/configuration.html>`_ for reference. 

Part 4. Result evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^

:raw-html-m2r:`<img src='https://github.com/Huangying-Zhan/DF-VO/blob/docs/docs/source/misc/dfvo_result.png?raw=true' width=320 height=480>`

:raw-html-m2r:`<img src='https://github.com/Huangying-Zhan/DF-VO/blob/docs/docs/source/misc/dfvo_result2.png?raw=true' width=400 height=100>`

The original results, including related works, can be found `here <https://www.dropbox.com/sh/u7x3rt4lz6zx8br/AADshjd33Q3TLCy2stKt6qpJa?dl=0>`_.

KITTI
~~~~~

`KITTI Odometry benchmark <http://www.cvlibs.net/datasets/kitti/eval_odometry.php>`_ contains 22 stereo sequences, in which 11 sequences are provided with ground truth. The 11 sequences are used for evaluating visual odometry. 

.. code-block::

   python tools/evaluation/odometry/eval_odom.py --result result/tmp/0 --align 6dof

For more information about the evaluation toolkit, please check the `toolbox page <https://github.com/Huangying-Zhan/kitti_odom_eval>`_ or the `wiki page <https://github.com/Huangying-Zhan/DF-VO/wiki>`_.

Part 5. Run your own dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also provide a guideline to run DF-VO on your own dataset.
Please check `Run own dataset <https://df-vo.readthedocs.io/en/latest/rsts/run_own_dataset.html>`_ for more details.

License
^^^^^^^

For academic usage, the code is released under the permissive MIT license. Our intension of sharing the project is for research/personal purpose. For any commercial purpose, please contact the authors. 

Acknowledgement
^^^^^^^^^^^^^^^

Some of the codes were borrowed from the excellent works of `monodepth2 <https://github.com/nianticlabs/monodepth2>`_\ , `LiteFlowNet <https://github.com/twhui/LiteFlowNet>`_ and `pytorch-liteflownet <https://github.com/sniklaus/pytorch-liteflownet>`_. The borrowed files are licensed under their original license respectively.
