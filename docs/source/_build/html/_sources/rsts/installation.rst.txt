============
Installation
============

This code was tested with Python 3.6, CUDA 9.0, Ubuntu 16.04, and PyTorch_.


We suggest use Anaconda_ for installing the prerequisites.

.. code-block:: shell

    # clone the project
    git clone https://github.com/Huangying-Zhan/DF-VO.git
    
    # create conda environment and install prerequisites
    cd DF-VO/envs
    conda env create -f requirement.yml -p dfvo 

    # activate the environment [dfvo]
    conda activate dfvo  


.. _PyTorch: https://pytorch.org/
.. _Anaconda: https://www.anaconda.com/distribution/