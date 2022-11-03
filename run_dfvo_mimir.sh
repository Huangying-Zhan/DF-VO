# MIMIR_tracks='SeaFloor0 SeaFloor1 SeaFloor_Algae_0 SeaFloor_Algae_1 OceanFloor0light OceanFloor0dark OceanFloor0light'
MIMIR_tracks='SeaFloor1 SeaFloor_Algae_0 SeaFloor_Algae_1 OceanFloor0light OceanFloor0dark OceanFloor0light'

for track in $MIMIR_tracks
do

    python apis/run.py \
    -d options/examples/default_configuration.yml \
    -c options/examples/mimir/mimir_tracking_$track.yml \
    --no_confirm

done