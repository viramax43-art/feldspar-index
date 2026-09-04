#!/usr/bin/env bash
set -euo pipefail

# systemd unit for Voicer Telegram bot
UNIT=/etc/systemd/system/voicer.service
echo '762341Aa@@' | sudo -S tee "$UNIT" >/dev/null <<'EOF'
[Unit]
Description=Voicer Telegram voice bot (GigaChat + XTTS/Silero)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=debian
Group=debian
WorkingDirectory=/home/debian/voicer
Environment=PYTHONUNBUFFERED=1
Environment=HOME=/home/debian
ExecStart=/home/debian/voicer/run_bot.sh
Restart=always
RestartSec=10
# XTTS first download/load can take several minutes on CPU
TimeoutStartSec=0
# Give the process enough file descriptors / memory headroom
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

echo '762341Aa@@' | sudo -S systemctl daemon-reload
echo '762341Aa@@' | sudo -S systemctl enable voicer.service
echo '762341Aa@@' | sudo -S systemctl restart voicer.service
sleep 3
echo '762341Aa@@' | sudo -S systemctl status voicer.service --no-pager -l | head -40
echo '--- logs ---'
echo '762341Aa@@' | sudo -S journalctl -u voicer.service -n 40 --no-pager
