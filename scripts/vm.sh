#!/usr/bin/env bash
# The Phase 3 server: one always-free e2-micro (us-central1) running `tradeengine run` 24/7 on the
# Binance DEMO account. Run from your Mac:
#
#   ./scripts/vm.sh create            once: service account, backup bucket, firewall, the VM
#   ./scripts/vm.sh deploy            committed code -> server; keys read from Secret Manager ONCE and
#                                     written to a root-only file on the server; service restarted
#   ./scripts/vm.sh te <command>      run a tradeengine command on the server, e.g. `te status`,
#                                     `te telegram-setup`, `te kill "reason"`, `te settings`
#   ./scripts/vm.sh apply <result.json> <strategy>   switch on one strategy from a holdout result
#   ./scripts/vm.sh logs              last 200 lines of the engine's log
#   ./scripts/vm.sh backup            run the nightly backup now
#   ./scripts/vm.sh ssh               a shell on the server
#
# Security: SSH only through Google's IAP tunnel (no open SSH port); the VM's service account can only
# CREATE backup files in one bucket, and cannot read Secret Manager. Keys never touch your disk.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${GCP_PROJECT:-televita-trading}"
REGION="us-central1"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-trading-engine}"
SA_NAME="trading-vm"
SA="$SA_NAME@$PROJECT.iam.gserviceaccount.com"
BUCKET="$PROJECT-backups"
SERVER_SECRETS="BINANCE_API_KEY BINANCE_API_SECRET TELEGRAM_BOT_TOKEN"

g() { gcloud --project="$PROJECT" --quiet "$@"; }
remote() { g compute ssh "$VM" --zone="$ZONE" --tunnel-through-iap --command="$1"; }

wait_for_ssh() {
  for _ in $(seq 1 30); do
    if remote "true" >/dev/null 2>&1; then return 0; fi
    sleep 10
  done
  echo "The server did not accept SSH within 5 minutes." >&2
  exit 1
}

cmd_create() {
  echo "== APIs =="
  g services enable compute.googleapis.com iap.googleapis.com storage.googleapis.com
  echo "== Service account for the VM (no roles; only bucket write below) =="
  g iam service-accounts describe "$SA" >/dev/null 2>&1 \
    || g iam service-accounts create "$SA_NAME" --display-name="Trading engine VM (backups only)"
  echo "== Backup bucket: $BUCKET ($REGION, private, files deleted after 30 days) =="
  if ! g storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
    g storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access --public-access-prevention
    printf '{"rule":[{"action":{"type":"Delete"},"condition":{"age":30}}]}' > /tmp/te-lifecycle.json
    g storage buckets update "gs://$BUCKET" --lifecycle-file=/tmp/te-lifecycle.json
    rm -f /tmp/te-lifecycle.json
  fi
  g storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$SA" --role=roles/storage.objectCreator >/dev/null
  g compute networks describe default >/dev/null 2>&1 || g compute networks create default --subnet-mode=auto
  echo "== Firewall: SSH only from Google's IAP range; no other inbound =="
  g compute firewall-rules describe allow-ssh-from-iap >/dev/null 2>&1 \
    || g compute firewall-rules create allow-ssh-from-iap --network=default --direction=INGRESS \
         --allow=tcp:22 --source-ranges=35.235.240.0/20
  for rule in default-allow-ssh default-allow-rdp; do
    g compute firewall-rules describe "$rule" >/dev/null 2>&1 && g compute firewall-rules delete "$rule"
  done
  echo "== VM: $VM (e2-micro, 30 GB standard disk = always-free tier) =="
  if ! g compute instances describe "$VM" --zone="$ZONE" >/dev/null 2>&1; then
    g compute instances create "$VM" --zone="$ZONE" --machine-type=e2-micro \
      --image-family=debian-12 --image-project=debian-cloud \
      --boot-disk-size=30GB --boot-disk-type=pd-standard \
      --service-account="$SA" --scopes=cloud-platform \
      --metadata=enable-oslogin=TRUE --shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring
  fi
  wait_for_ssh
  echo "Server ready. Next: ./scripts/vm.sh deploy"
}

cmd_deploy() {
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "You have uncommitted changes; only COMMITTED code is deployed. Commit first (or stash) and run again." >&2
    exit 1
  fi
  SHA="$(git rev-parse --short HEAD)"
  echo "== Uploading commit $SHA =="
  git archive --format=tar.gz -o "/tmp/tradeengine-$SHA.tar.gz" HEAD
  g compute scp "/tmp/tradeengine-$SHA.tar.gz" "$VM:/tmp/" --zone="$ZONE" --tunnel-through-iap
  rm -f "/tmp/tradeengine-$SHA.tar.gz"

  echo "== Keys: read from Secret Manager now, sent straight to a root-only file on the server =="
  ENV_TEXT="TRADING_MODE=demo
DATABASE_URL=sqlite+aiosqlite:////var/lib/tradeengine/engine.db
DATA_DIR=/var/lib/tradeengine/data"
  for NAME in $SERVER_SECRETS; do
    VALUE="$(gcloud secrets versions access latest --secret="$NAME" --project="$PROJECT")"
    ENV_TEXT="$ENV_TEXT
$NAME=$VALUE"
  done
  unset VALUE
  printf '%s\n' "$ENV_TEXT" | g compute ssh "$VM" --zone="$ZONE" --tunnel-through-iap \
    --command="sudo install -d -m 700 /etc/tradeengine && sudo sh -c 'umask 077; cat > /etc/tradeengine/env' && sudo chmod 600 /etc/tradeengine/env" \
    -- -T  # no terminal: the keys arrive on stdin byte for byte, never on a command line
  unset ENV_TEXT

  echo "== Install and restart =="
  remote "set -e; sudo mkdir -p /opt/tradeengine/releases/$SHA && sudo tar -xzf /tmp/tradeengine-$SHA.tar.gz -C /opt/tradeengine/releases/$SHA \
    && rm -f /tmp/tradeengine-$SHA.tar.gz && sudo bash /opt/tradeengine/releases/$SHA/deploy/setup.sh /opt/tradeengine/releases/$SHA $BUCKET \
    && sudo systemctl restart tradeengine && sleep 20 && systemctl is-active tradeengine && sudo te status"
  echo "Deployed $SHA."
}

cmd_te() { remote "sudo te $(printf '%q ' "$@")"; }

cmd_apply() {
  [ $# -eq 2 ] || { echo "usage: ./scripts/vm.sh apply <holdout result.json> <strategy>" >&2; exit 1; }
  g compute scp "$1" "$VM:/tmp/holdout-result.json" --zone="$ZONE" --tunnel-through-iap
  remote "sudo install -o trader -m 640 /tmp/holdout-result.json /var/lib/tradeengine/data/holdout-result.json && rm /tmp/holdout-result.json \
    && sudo te gate2-apply /var/lib/tradeengine/data/holdout-result.json --only $(printf '%q' "$2") --dry-run \
    && sudo te gate2-apply /var/lib/tradeengine/data/holdout-result.json --only $(printf '%q' "$2")"
}

cmd_logs() { remote "sudo journalctl -u tradeengine -n 200 --no-pager"; }
cmd_backup() { remote "sudo systemctl start tradeengine-backup && sudo journalctl -u tradeengine-backup -n 3 --no-pager"; }
cmd_ssh() { g compute ssh "$VM" --zone="$ZONE" --tunnel-through-iap; }

case "${1:-}" in
  create) cmd_create ;;
  deploy) cmd_deploy ;;
  te) shift; cmd_te "$@" ;;
  apply) shift; cmd_apply "$@" ;;
  logs) cmd_logs ;;
  backup) cmd_backup ;;
  ssh) cmd_ssh ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
