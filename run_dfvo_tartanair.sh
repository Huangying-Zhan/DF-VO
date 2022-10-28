tartanAIR_tracks='afsample'
for track in $tartanAIR_tracks
do

    python apis/run.py \
    -d options/examples/default_configuration.yml \
    -c options/examples/tartanair/tartanair_$track.yml \
    --no_confirm

done