============
Introduction
============

DF-VO is a frame-to-frame VO system integrating geometry and deep learning.


.. _project-structure:

-----------------
Project Structure
-----------------

    This project has the following structure:

    .. code-block:: shell

        - DF-VO
            - apis                              # APIs
            - dataset                           # dataset directories
            - envs                              # requirement files to create anaoncda envs
            - docs                              # documentation
            - libs                              # packages
                - dfvo.py                       # core DF-VO program
                - datasets                      # dataset loaders
                - deep_models                   # deep networks
                    - depth                     # depth models
                    - flow                      # optical flow models
                    - pose                      # ego-motion models
                - geometry                      # geometry related operations
                - matching                      # feature matching related packages
                - tracker                       # tracker packages
                - general                       # general functions
            - model_zoo                         # pretrained deep models
            - options                           # configurations
            - scripts                           # scripts for running experiments
            - tools                             # experiment tools, e.g. evaluation
