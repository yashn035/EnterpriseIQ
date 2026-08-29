#!/usr/bin/env bash
set -e

# Configuration variables (override via environment variables as needed)
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-123456789012}"
ECR_REPO_NAME="${ECR_REPO_NAME:-enterprise-iq-rag}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "=================================================="
echo " EnterpriseIQ AWS ECR & EC2 Deployment Orchestrator "
echo "=================================================="
echo "AWS Region:      ${AWS_REGION}"
echo "AWS Account ID:  ${AWS_ACCOUNT_ID}"
echo "ECR Repository:  ${ECR_REPO_NAME}"
echo "Image Tag:       ${IMAGE_TAG}"
echo "ECR URI:         ${ECR_URI}"
echo "=================================================="

# Step 1: Build Docker Image
echo "[1/4] Building Docker image '${ECR_REPO_NAME}:${IMAGE_TAG}'..."
docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}" .

# Step 2: Authenticate AWS CLI with ECR
echo "[2/4] Authenticating AWS CLI with ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Ensure ECR repository exists
aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${ECR_REPO_NAME}" --region "${AWS_REGION}"

# Step 3: Tag and Push Image to ECR
echo "[3/4] Tagging and pushing Docker image to ECR..."
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${IMAGE_TAG}"

# Step 4: Display AWS EC2 deployment command
echo "=================================================="
echo "[4/4] Image successfully pushed to ECR!"
echo "=================================================="
echo ""
echo "To deploy this container to an AWS EC2 t2.medium instance, run:"
echo ""
cat << EOF
aws ec2 run-instances \\
    --image-id ami-0c7217cdde317cfec \\
    --instance-type t2.medium \\
    --key-name your-ec2-keypair \\
    --security-group-ids sg-xxxxxx \\
    --subnet-id subnet-xxxxxx \\
    --user-data '#!/bin/bash
sudo yum update -y
sudo amazon-linux-extras install docker -y || sudo yum install docker -y
sudo systemctl start docker
sudo systemctl enable docker
sudo yum install aws-cli -y
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
docker run -d --restart always -p 8000:8000 ${ECR_URI}:${IMAGE_TAG}
'
EOF
