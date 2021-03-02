============
Introduction
============

DF-VO is a frame-to-frame VO system integrating geometry and deep learning.


.. _project-structure:

-----------------
Project Structure
-----------------

    This project has the following structure

    .. code-block:: shell

        - DF-VO
            - apis          # APIs
            - dataset       # dataset directories
            - docs          # documentation
            - libs          # packages
            - model_zoo     # pretrained models
            - options       # configurations
            - scripts       # scripts for running experiments
            - tool          # experiment tools, e.g. evaluation

--------
Examples
--------

.. _run-dfvo:

    .. code-block:: python

        # Example 1: run default kitti setup
        python run.py -d options/kitti_default_configuration.yml  

        # Example 2: Run custom kitti setup
        # kitti_default_configuration.yml and kitti_stereo_0.yml are merged
        python run.py -c options/kitti/kitti_stereo_0.yml  

