#!/bin/sh
# Standalone: no FeatureCloud controller / supervisord / nginx.
# cwd must be / so relative mnt/input resolves to the FC mount at /mnt/input.
standalone="$(printf '%s' "${STANDALONE:-}" | tr '[:upper:]' '[:lower:]')"
case "$standalone" in
  1|true|yes)
    cd /
    exec python3 -u /app/main.py
    ;;
esac

/usr/bin/supervisord -c "/supervisord.conf"
