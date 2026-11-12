#!/usr/bin/env bash
# Nightly: a consistent copy of the SQLite journal, uploaded to the backup bucket with the VM's own
# service account (it may only CREATE objects in that bucket; it has no other permissions).
set -euo pipefail
BUCKET="$1"
DB=/var/lib/tradeengine/engine.db
OUT=/var/lib/tradeengine/backup.db
[ -f "$DB" ] || { echo "no database yet"; exit 0; }
# SQLite's online backup API: a consistent copy even while the engine writes (WAL mode)
python3 -c 'import sqlite3, sys; src = sqlite3.connect(sys.argv[1]); dst = sqlite3.connect(sys.argv[2]); src.backup(dst); dst.close()' "$DB" "$OUT"
TOKEN=$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
NAME="engine-$(date -u +%Y%m%d-%H%M).db"
curl -sf -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/octet-stream' --data-binary @"$OUT" \
  "https://storage.googleapis.com/upload/storage/v1/b/$BUCKET/o?uploadType=media&name=$NAME" >/dev/null
rm -f "$OUT"
echo "backed up to gs://$BUCKET/$NAME"
