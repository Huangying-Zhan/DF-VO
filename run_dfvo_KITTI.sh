KITTI_tracks='00 01 02 03 04 05 06 07 08 09 10'

for track in $KITTI_tracks
do

    python apis/run.py \
    -d options/examples/default_configuration.yml \
    -c options/examples/kitti/kitti_tracking_$track.yml \
    --no_confirm

done