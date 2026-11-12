#!/usr/bin/env bash
# Run one command with the demo keys loaded from Google Secret Manager.
# Keys exist only in this process's environment: nothing is written to disk.
#
#   ./scripts/with-secrets.sh uv run tradeengine status
#
# Project defaults to televita-trading; override with GCP_PROJECT=...
set -euo pipefail

PROJECT="${GCP_PROJECT:-televita-trading}"
NAMES="CAPITAL_API_KEY CAPITAL_IDENTIFIER CAPITAL_API_PASSWORD BINANCE_API_KEY BINANCE_API_SECRET ANTHROPIC_API_KEY TELEGRAM_BOT_TOKEN"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found. Install the Google Cloud CLI first." >&2
  exit 1
fi

loaded=""
for NAME in $NAMES; do
  if VALUE="$(gcloud secrets versions access latest --secret="$NAME" --project="$PROJECT" 2>/dev/null)"; then
    export "$NAME=$VALUE"
    loaded="$loaded $NAME"
  fi
done
unset VALUE

if [ -z "$loaded" ]; then
  echo "No secrets could be read from project $PROJECT. Check 'gcloud auth login' and the project name." >&2
  exit 1
fi
echo "Loaded from Secret Manager:$loaded" >&2

exec "$@"
