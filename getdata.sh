aws2 s3 sync s3://voice-antispoofing/protocol_V2 s3data/protocol_V2
aws2 s3 sync --exclude '*/eval-files/*' s3://voice-antispoofing/wideband-768 s3data/wideband-768
#aws2 s3 sync --exclude '*/eval-files/*' s3://voice-antispoofing/narrow-wide s3data/narrow-wide
