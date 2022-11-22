TUM_tracks='freiburg3_nostructure_notexture_far'

for track in $TUM_tracks
do

    python apis/run.py \
    -d options/examples/default_configuration.yml \
    -c options/examples/tum/$track.yml \
    --no_confirm

done