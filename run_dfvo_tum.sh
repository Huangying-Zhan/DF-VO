TUM_tracks='freiburg1_360 freiburg1_rpy freiburg3_nostructure_notexture_far freiburg3_nostructure_texture_near_withloop'

for track in $TUM_tracks
do

    python apis/run.py \
    -d options/examples/default_configuration.yml \
    -c options/examples/tum/$track.yml \
    --no_confirm

done