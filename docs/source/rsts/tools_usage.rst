
-----------
Tools usage
-----------

.. _tools_usage:

^^^^^^^^^^
Evaluation
^^^^^^^^^^

.. _evaluation:

    To evaluate the odometry result on KITTI dataset, here we provide an example.
    For details, please refer to the eval_odom_ wiki page. 

    .. code-block:: shell

        # Evaluate Odometry Split
        python tools/evaluation/eval_odom.py \
        --result {RESULT_DIR} \
        --gt dataset/kitti_odom/gt_poses/ \
        --align 7dof \ 
        --seqs "09" "10"
        

^^^^^^^^^^^^^
General tools
^^^^^^^^^^^^^

.. _general_tools:

    .. code-block:: shell

        # Generate ground truth poses from KITTI Raw dataset
        python tools/generate_kitti_raw_pose.py \
        --data_dir dataset/kitti_raw \
        --result_dir dataset/kitti_raw_pose \
        --seqs 2011_09_26_drive_0005_sync 2011_09_26_drive_0009_sync

        # Generate KITTI Flow 2012/2015 prediction
        python tools/generate_flow_prediction.py \
        --dataset kitti2012 \
        --model {FLOW_MODEL_PATH} \
        --result {RESULT_DIR}

.. _eval_odom: https://github.com/Huangying-Zhan/DF-VO/wiki/eval_odom