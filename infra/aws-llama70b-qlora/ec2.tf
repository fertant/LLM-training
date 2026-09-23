# AWS Deep Learning AMI (Ubuntu) with NVIDIA drivers, CUDA, and a
# preinstalled PyTorch conda env - avoids driver-install headaches on a
# fresh GPU instance.
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch * (Ubuntu 22.04) *"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

locals {
  user_data = templatefile("${path.module}/scripts/user_data.sh.tpl", {
    aws_region        = var.aws_region
    bucket_name       = aws_s3_bucket.assets.bucket
    hf_token_ssm_name = aws_ssm_parameter.hf_token.name
    base_model_id     = var.base_model_id
    num_train_epochs  = var.num_train_epochs
  })
}

resource "aws_instance" "finetune" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.instance.id]
  iam_instance_profile   = aws_iam_instance_profile.instance.name
  key_name               = var.key_pair_name
  user_data              = local.user_data

  root_block_device {
    volume_size = var.root_volume_size_gb
    volume_type = "gp3"
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price                      = var.spot_max_price
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  tags = {
    Name    = var.project_name
    Project = var.project_name
  }
}
