#!/usr/bin/env bash
set -euo pipefail
project=$(cd "$(dirname "$0")/.." && pwd)
for role in benchmark web; do
cat > "/etc/supervisor/conf.d/voicehub-arena-$role.conf" <<EOF
[program:voicehub-arena-$role]
directory=$project
command=/bin/bash "$project/scripts/service.sh" $role
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stopwaitsecs=30
redirect_stderr=true
stdout_logfile=$project/$role.log
stdout_logfile_maxbytes=5MB
stdout_logfile_backups=2
EOF
done
supervisorctl reread
supervisorctl update
echo 'Services installed and stopped. Start explicitly with:'
echo 'supervisorctl start voicehub-arena-benchmark voicehub-arena-web'
