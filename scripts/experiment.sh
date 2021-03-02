# Reference Model
python apis/run.py -d options/examples/default_configuration.yml  

# Stereo Train (ICRA)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/kitti_stereo_train_icra.yml \
--no_confirm

# Mono-SC Train (ICRA)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/kitti_mono_sc_train_icra.yml \
--no_confirm

# Stereo Train (extended)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/kitti_stereo_train_extend.yml \
--no_confirm

# Mono-SC Train (extended)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/kitti_mono_sc_train_extend.yml \
--no_confirm

############## Ablation study experiments (extended paper) ##############
# Tracker (PnP)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/ablation_tracker_pnp.yml \
--no_confirm

# Self-Flow (Offline)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/ablation_self_flow_offline.yml \
--no_confirm

# Self-Flow (Online)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/ablation_self_flow_online.yml \
--no_confirm

# Depth (Mono-SC)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/ablation_depth_mono_sc.yml \
--no_confirm

# Depth (Mono)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/ablation_depth_mono.yml \
--no_confirm

# Correspondences (Uniform)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/ablation_correspondences_uniform.yml \
--no_confirm

# Correspondences (Best-N)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/ablation_correspondences_best_n.yml \
--no_confirm

# Scale (Iterative)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/ablation_scale_iterative.yml \
--no_confirm

# Model Selection (Flow)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/ablation_model_sel_flow.yml \
--no_confirm

# Image Resolution (Full)
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/ablation_img_res_full.yml \
--no_confirm

############## KITTI Tracking sequences experiments (extended paper) ##############
# KITTI Tracking
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/kitti_tracking.yml \
--no_confirm

# Oxford robotcar
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/oxford_robotcar.yml \
--no_confirm

# TUM RGBD SLAM
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/tum_rgbd_slam.yml \
--no_confirm

# Adelaide driving sequence
python apis/run.py \
-d options/examples/default_configuration.yml \
-c options/examples/adelaide_driving.yml \
--no_confirm
