data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "instance" {
  name        = "${var.project_name}-sg"
  description = "Fine-tuning instance SG - egress only by default, SSH ingress only if a key pair is supplied."
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "Allow all outbound (model/package downloads, S3, SSM)."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project = var.project_name
  }
}

resource "aws_security_group_rule" "ssh" {
  count = var.key_pair_name != null ? 1 : 0

  type              = "ingress"
  from_port         = 22
  to_port           = 22
  protocol          = "tcp"
  cidr_blocks       = [var.allowed_ssh_cidr]
  security_group_id = aws_security_group.instance.id
  description       = "SSH access - narrow allowed_ssh_cidr to your own IP."
}
