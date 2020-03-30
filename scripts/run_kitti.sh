############# Run particular exp #############
python run.py -c options/kitti/sampling_test.yml 
# python run.py -c options/kitti/sampling_test.yml --no_confirm
# python run.py -c options/kitti/dfvo_test.yml --no_confirm
# python run.py -c options/kitti/dfvo_test.yml 

############# Run odom seqs #############
# for seq in \
# 9 10 4 7 0 1 2 3 5 6 8 
# # 1 2 3 5 6 8 
# # 9 10
# # 2 3 5 6 7 8
# do
#     python run.py -s $seq -c options/kitti/sampling_test.yml --no_confirm
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
#     python run.py -s $seq -c options/kitti/sampling_test.yml --no_confirm
#     # python run.py -s $seq -c options/kitti/dfvo_test.yml --no_confirm
# done



# # python run.py -c options/kitti/kitti_stereo_0.yml