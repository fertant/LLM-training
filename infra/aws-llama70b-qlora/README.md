# Llama 3.1/3.3 70B QLoRA fine-tuning on AWS

Provisions one GPU EC2 instance, uploads `../../training_data/` to S3, and
runs unattended QLoRA fine-tuning of a 70B Llama checkpoint on it.

- **Instance**: `g5.12xlarge` (4x A10G, 24GB each = 96GB total GPU memory).
  The base model is quantized to 4-bit (~35-40GB) and sharded across the 4
  GPUs; only small LoRA adapter matrices are trained on top, so this stays
  far cheaper than full fine-tuning (which needs 8x A100/H100 80GB).
- **Auth**: the Hugging Face token is stored as an SSM SecureString, never
  written into user_data or committed to state in plaintext beyond
  Terraform's normal state handling.
- **Output**: the fine-tuned LoRA adapter (a few hundred MB, not the full
  70B model) is synced to `s3://<bucket>/output/`.

## Cost warning

`g5.12xlarge` is roughly **$5.67/hr on-demand** (less with `use_spot = true`,
the default). A 70B QLoRA run against a small corpus like this repo's
`training_data/` should finish in well under an hour, but **you are
responsible for destroying this stack when done** - it does not
self-terminate. Run `terraform destroy` when finished, and check the AWS
console for the instance to confirm it's gone.

## Prerequisites

1. Accept the Llama license for `var.base_model_id` on Hugging Face and
   generate an access token with read access to it.
2. AWS credentials configured locally (`aws configure` or env vars) with
   permission to create EC2/S3/IAM/SSM resources.
3. `terraform >= 1.5`.

## Usage

```bash
cd infra/aws-llama70b-qlora
terraform init

# Never commit a .tfvars file containing the token - export it instead:
export TF_VAR_hf_token="hf_xxx..."

terraform plan
terraform apply
```

Watch progress once the instance is up:

```bash
terraform output tail_logs_command   # prints the ready-to-run aws ssm command
```

When training finishes, the adapter is already in
`s3://<bucket>/output/`; `terraform output download_adapter_command` gives
you the `aws s3 sync` to pull it down locally.

**When you're done, tear it down:**

```bash
terraform destroy
```

## Overriding defaults

- `instance_type` - bump to `p4de.24xlarge` (8x A100 80GB) if you want full
  fine-tuning instead of QLoRA (would also require rewriting
  `scripts/finetune_llama70b_qlora.py` to drop the 4-bit quantization and
  LoRA config).
- `base_model_id` - defaults to `meta-llama/Llama-3.1-70B-Instruct`; set to
  `meta-llama/Llama-3.3-70B-Instruct` (or any other) via
  `-var base_model_id=...`.
- `use_spot = false` - use On-Demand if you can't tolerate Spot
  interruption for this run.
