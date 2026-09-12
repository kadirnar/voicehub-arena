#!/usr/bin/env bash
set -euo pipefail
project=$(cd "$(dirname "$0")/.." && pwd)
for role in benchmark web; do
command="/bin/bash \"$project/scripts/service.sh\" $role"
logfile="$project/$role.log"
logsize=5MB
if [[ -f /opt/supervisor-scripts/utils/logging.sh && -f /opt/supervisor-scripts/utils/environment.sh ]]; then
  # Vast routes wrapper output to its portal; private SSH access needs no
  # portal entry or public port. Keep the image's management services intact.
  wrapper="/opt/supervisor-scripts/voicehub-arena-$role.sh"
  cat > "$wrapper" <<EOF
#!/usr/bin/env bash
source /opt/supervisor-scripts/utils/logging.sh
source /opt/supervisor-scripts/utils/environment.sh
exec /bin/bash "$project/scripts/service.sh" $role
EOF
  chmod +x "$wrapper"
  command="$wrapper"
  logfile=/dev/stdout
  logsize=0
fi
cat > "/etc/supervisor/conf.d/voicehub-arena-$role.conf" <<EOF
[program:voicehub-arena-$role]
environment=PROC_NAME="%(program_name)s"
directory=$project
command=$command
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stopwaitsecs=30
redirect_stderr=true
stdout_logfile=$logfile
stdout_logfile_maxbytes=$logsize
stdout_logfile_backups=2
EOF
done
supervisorctl reread
supervisorctl update
echo 'Services installed and stopped. Start explicitly with:'
echo 'supervisorctl start voicehub-arena-benchmark voicehub-arena-web'
