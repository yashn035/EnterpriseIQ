#!/bin/sh
PORT="${PORT:-10000}"
echo "Starting EnterpriseIQ on port $PORT..."
exec streamlit run app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
