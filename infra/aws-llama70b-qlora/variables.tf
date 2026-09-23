variable "aws_region" {
  description = "AWS region to provision the fine-tuning instance in. g5 instances aren't available in every region."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix used to name/tag every resource this stack creates."
  type        = string
  default     = "llama70b-qlora-finetune"
}

variable "instance_type" {
  description = <<-EOT
    EC2 GPU instance type. Default is g5.12xlarge (4x A10G, 24GB each = 96GB
    total GPU memory) which comfortably fits a 4-bit-quantized 70B model
    (~35-40GB) sharded across the 4 GPUs for QLoRA. Do not drop below this
    without re-checking the math - fewer/smaller GPUs will OOM.
  EOT
  type        = string
  default     = "g5.12xlarge"
}

variable "use_spot" {
  description = "Request a Spot instance instead of On-Demand. Much cheaper (~60-70% off) but can be interrupted - fine for a resumable fine-tuning job with checkpointing, riskier otherwise."
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Max hourly price (USD) to bid for the Spot instance. Leave null to default to the On-Demand price (i.e. never pay more than On-Demand)."
  type        = string
  default     = null
}

variable "root_volume_size_gb" {
  description = "Root EBS volume size in GB. 70B checkpoints, the quantized base model, and HF cache need real space."
  type        = number
  default     = 300
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair to allow SSH access. Leave null to skip creating an SSH ingress rule entirely (then use SSM Session Manager instead)."
  type        = string
  default     = null
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into the instance on port 22. Only used if key_pair_name is set. Narrow this to your own IP (e.g. \"203.0.113.4/32\") before applying - do not leave it open to the world."
  type        = string
  default     = "0.0.0.0/0"
}

variable "base_model_id" {
  description = "Hugging Face model id to fine-tune."
  type        = string
  default     = "meta-llama/Llama-3.1-70B-Instruct"
}

variable "hf_token" {
  description = <<-EOT
    Hugging Face access token with access to the gated Llama model (accept
    the license at huggingface.co first). Stored as a SecureString in SSM
    Parameter Store, never written into user_data or Terraform state in
    plaintext beyond the state file's normal handling of sensitive values.
    Pass it via TF_VAR_hf_token or a .tfvars file that is gitignored.
  EOT
  type        = string
  sensitive   = true
}

variable "training_data_dir" {
  description = "Local path to the training data folder to upload to S3 and fine-tune on."
  type        = string
  default     = "../../training_data"
}

variable "num_train_epochs" {
  description = "Number of fine-tuning epochs."
  type        = number
  default     = 3
}
