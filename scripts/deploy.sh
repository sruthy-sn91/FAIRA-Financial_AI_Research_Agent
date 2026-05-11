#!/bin/bash
# =============================================================================
# Deployment Script
# =============================================================================
# Run from your LOCAL machine to deploy the latest code to EC2.
# This is what GitHub Actions will call in Phase 10.
#
# Usage:
#   chmod +x scripts/deploy.sh
#   EC2_HOST=<your-ec2-ip> ./scripts/deploy.sh
# =============================================================================

set -e

EC2_HOST="${EC2_HOST:?Error: EC2_HOST environment variable is required}"
EC2_USER="${EC2_USER:-ubuntu}"
EC2_KEY="${EC2_KEY:-~/.ssh/financial-agent-key.pem}"
REMOTE_DIR="/home/ubuntu/financial-agent"

echo "=== Deploying to $EC2_USER@$EC2_HOST ==="

# ── 1. Push latest code via git ───────────────────────────────────────────────
echo "[1/4] Pulling latest code on server..."
ssh -i "$EC2_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" \
    "cd $REMOTE_DIR && git pull origin main"

# ── 2. Rebuild Docker image ───────────────────────────────────────────────────
echo "[2/4] Rebuilding Docker image..."
ssh -i "$EC2_KEY" "$EC2_USER@$EC2_HOST" \
    "cd $REMOTE_DIR && docker compose build --no-cache"

# ── 3. Restart services ───────────────────────────────────────────────────────
echo "[3/4] Restarting services..."
ssh -i "$EC2_KEY" "$EC2_USER@$EC2_HOST" \
    "cd $REMOTE_DIR && docker compose --profile prod up -d"

# ── 4. Health check ───────────────────────────────────────────────────────────
echo "[4/4] Verifying deployment..."
sleep 10  # Give services time to start

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    "http://$EC2_HOST/health" || echo "000")

if [ "$HTTP_STATUS" = "200" ]; then
    echo ""
    echo "=== Deployment successful! ==="
    echo "  UI:     http://$EC2_HOST"
    echo "  API:    http://$EC2_HOST/api/docs"
    echo "  MLflow: http://$EC2_HOST/mlflow"
else
    echo ""
    echo "=== WARNING: Health check returned HTTP $HTTP_STATUS ==="
    echo "Check logs: ssh -i $EC2_KEY $EC2_USER@$EC2_HOST 'cd $REMOTE_DIR && docker compose logs --tail=50'"
    exit 1
fi
