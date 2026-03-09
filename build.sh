#!/usr/bin/env bash
set -e

echo "=== Building SU Admin Classifier ==="

# 1. Python dependencies
echo "--- Installing Python dependencies ---"
pip install -r backend/requirements-deploy.txt

# 2. Frontend build
echo "--- Building React frontend ---"
cd frontend
npm install
npm run build
cd ..

# 3. Copy built frontend to backend/static (Flask serves it)
echo "--- Deploying frontend to backend/static ---"
rm -rf backend/static
cp -r frontend/dist backend/static

echo "=== Build complete ==="
