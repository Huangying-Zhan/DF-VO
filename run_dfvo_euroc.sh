EUROC_tracks='MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult'

for track in $EUROC_tracks
do

    python apis/run.py \
    -d options/examples/default_configuration.yml \
    -c options/examples/euroc/euroc_tracking_$track.yml \
    --no_confirm

done