AQUALOC_tracks='Archaeological_site_sequences/archaeo_sequence_1_raw_data'
# AQUALOC_tracks='Archaeological_site_sequences/archaeo_sequence_2_raw_data Archaeological_site_sequences/archaeo_sequence_3_raw_data Archaeological_site_sequences/archaeo_sequence_4_raw_data Archaeological_site_sequences/archaeo_sequence_5_raw_data Archaeological_site_sequences/archaeo_sequence_6_raw_data'

for track in $AQUALOC_tracks
do

    python apis/run.py \
    -d options/examples/default_configuration.yml \
    -c options/examples/aqualoc/$track.yml \
    --no_confirm

done