#!/bin/bash
# Bootstraps the instance and kicks off QLoRA fine-tuning unattended.
# Logs land in /var/log/finetune.log so you can `tail -f` it over SSM/SSH.
set -euxo pipefail

exec > >(tee -a /var/log/finetune.log) 2>&1
echo "=== Bootstrap started at $(date -u) ==="

WORKDIR=/home/ubuntu/finetune
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# The Deep Learning AMI ships a conda "pytorch" env with CUDA + torch preinstalled.
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

pip install --upgrade "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" \
    "accelerate>=0.33" datasets pypdf openpyxl python-docx

# Fetch the HF token from SSM (SecureString) rather than embedding it here.
export HF_TOKEN
HF_TOKEN=$(aws ssm get-parameter \
  --name "${hf_token_ssm_name}" \
  --with-decryption \
  --region "${aws_region}" \
  --query "Parameter.Value" --output text)

aws s3 sync "s3://${bucket_name}/training_data" "$WORKDIR/training_data"
aws s3 cp "s3://${bucket_name}/scripts/finetune_llama70b_qlora.py" "$WORKDIR/finetune_llama70b_qlora.py"

python "$WORKDIR/finetune_llama70b_qlora.py" \
  --base_model_id "${base_model_id}" \
  --training_data_dir "$WORKDIR/training_data" \
  --output_dir "$WORKDIR/output" \
  --num_train_epochs "${num_train_epochs}"

aws s3 sync "$WORKDIR/output" "s3://${bucket_name}/output"

echo "=== Bootstrap + fine-tuning finished at $(date -u) ==="
