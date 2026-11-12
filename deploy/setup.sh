#!/usr/bin/env bash
# Runs ON THE SERVER as root (scripts/vm.sh calls it after each upload). Idempotent.
# Installs uv, a 1 GB swap file, the `trader` user, the systemd service, the nightly backup timer
# and the `te` helper. Secrets are not handled here: vm.sh writes /etc/tradeengine/env separately.
set -euo pipefail
RELEASE="$1"            # /opt/tradeengine/releases/<git sha>
BUCKET="$2"             # backup bucket name, without gs://

id trader >/dev/null 2>&1 || useradd --system --home-dir /var/lib/tradeengine --create-home --shell /usr/sbin/nologin trader
install -d -o trader -g trader -m 750 /var/lib/tradeengine /var/lib/tradeengine/data /opt/tradeengine/python /opt/tradeengine/cache
install -d -o root -g root -m 700 /etc/tradeengine
apt-get -qq update >/dev/null && apt-get -qq install -y sqlite3 curl >/dev/null

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh >/dev/null
fi

if [ ! -f /swapfile ]; then  # e2-micro has 1 GB of memory; swap keeps a pandas spike from killing the engine
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

cat > /etc/tradeengine/uv.env <<UVENV
UV_PYTHON_INSTALL_DIR=/opt/tradeengine/python
UV_CACHE_DIR=/opt/tradeengine/cache
UV_PROJECT_ENVIRONMENT=/var/lib/tradeengine/venv
UVENV
chmod 644 /etc/tradeengine/uv.env

chown -R trader:trader "$RELEASE"
ln -sfn "$RELEASE" /opt/tradeengine/current
sudo -u trader env $(cat /etc/tradeengine/uv.env | xargs) /usr/local/bin/uv sync --frozen --no-dev --python 3.12 \
  --directory /opt/tradeengine/current >/dev/null

install -m 755 "$RELEASE/deploy/te" /usr/local/bin/te
install -m 755 "$RELEASE/deploy/backup.sh" /usr/local/bin/tradeengine-backup
sed "s/__BUCKET__/$BUCKET/" "$RELEASE/deploy/tradeengine-backup.service" > /etc/systemd/system/tradeengine-backup.service
install -m 644 "$RELEASE/deploy/tradeengine-backup.timer" /etc/systemd/system/tradeengine-backup.timer
install -m 644 "$RELEASE/deploy/tradeengine.service" /etc/systemd/system/tradeengine.service
systemctl daemon-reload
systemctl enable --quiet tradeengine.service tradeengine-backup.timer
systemctl start tradeengine-backup.timer
# keep the five newest releases
ls -1dt /opt/tradeengine/releases/* | tail -n +6 | xargs -r rm -rf
echo "setup done: $(readlink /opt/tradeengine/current)"
