############# Run particular exp #############
python run.py -d result/dfvo2/tro_paper/0/0/configuration_09.yml -c options/adelaide/default.yml  --no_confirm
# python run.py -d result/dfvo2/tro_paper/0/0/configuration_09.yml -c options/kitti/tro_exp/reference.yml  --no_confirm
# python run.py -c options/kitti/sampling_test.yml --no_confirm
# python run.py -c options/kitti/dfvo_test.yml --no_confirm
# python run.py -c options/kitti/dfvo_test.yml 

############# Run Adelaide seqs #############
# cam2
# for seq in \
# "00" "01" "02"
# do
#     python run.py -s $seq -d result/dfvo2/tro_paper/0/0/configuration_09.yml -c options/adelaide/exp0_1.yml  --no_confirm
#     python run.py -s $seq -d result/dfvo2/tro_paper/0/0/configuration_09.yml -c options/adelaide/exp1_1.yml  --no_confirm
#     python run.py -s $seq -d result/dfvo2/tro_paper/0/0/configuration_09.yml -c options/adelaide/exp0_2.yml  --no_confirm
#     python run.py -s $seq -d result/dfvo2/tro_paper/0/0/configuration_09.yml -c options/adelaide/exp2_1.yml  --no_confirm
#     python run.py -s $seq -d result/dfvo2/tro_paper/0/0/configuration_09.yml -c options/adelaide/exp2_2.yml  --no_confirm
# done

# # cam1
# for seq in \
# "00", "01", "02", "03"
# do
#     python run.py -s $seq -d result/dfvo2/tro_paper/0/0/configuration_09.yml -c options/adelaide/exp2_0.yml  --no_confirm
#     # python run.py -s $seq -c options/kitti/dfvo_test.yml --no_confirm
# done

############# Run odom seqs #############
# for seq in \
# 0 1 2 3 4 5 6 7 8 
# # 9 10
# # 9 10 4 7 0 1 2 3 5 6 8 
# # # 2 3 5 6 7 8
# do
    # python run.py -s $seq -d result/dfvo2/tro_paper/0/0/configuration_09.yml -c options/kitti/tro_exp/reference.yml  --no_confirm
#     # python run.py -s $seq -c options/kitti/dfvo_test.yml --no_confirm
# done

############# Run tracking seqs #############
# for seq in \
# 2011_09_26_drive_0005_sync \
# 2011_09_26_drive_0009_sync \
# 2011_09_26_drive_0011_sync \
# 2011_09_26_drive_0013_sync \
# 2011_09_26_drive_0014_sync \
# 2011_09_26_drive_0018_sync \
# 2011_09_29_drive_0004_sync \
# 2011_09_26_drive_0015_sync \
# 2011_10_03_drive_0047_sync \

# do
#     python run.py -s $seq -c options/kitti/tro_exp/reference.yml --no_confirm
#     # python run.py -s $seq -c options/kitti/sampling_test.yml --no_confirm
#     # python run.py -s $seq -c options/kitti/dfvo_test.yml --no_confirm
# done



# # python run.py -c options/kitti/kitti_stereo_0.yml