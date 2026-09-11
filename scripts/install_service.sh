#!/usr/bin/env bash
set -euo pipefail
project=$(cd "$(dirname "$0")/.." && pwd)
cat > /etc/supervisor/conf.d/voicehub-arena.conf <<EOF
[program:voicehub-arena]
directory=$project
command=$project/.venv/bin/voicehub-arena serve --runs $project/runs --port 7860
autostart=true
autorestart=unexpected
redirect_stderr=true
stdout_logfile=$project/server.log
stdout_logfile_maxbytes=5MB
stdout_logfile_backups=2
EOF
supervisorctl reread
supervisorctl update
supervisorctl status voicehub-arena
