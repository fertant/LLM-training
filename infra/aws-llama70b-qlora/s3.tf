resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "assets" {
  bucket = "${var.project_name}-${random_id.suffix.hex}"

  # This bucket only ever holds a copy of training_data/ and the fine-tuning
  # script for one run - force_destroy so `terraform destroy` doesn't get
  # stuck on non-empty-bucket errors after the instance writes checkpoints
  # back here.
  force_destroy = true

  tags = {
    Project = var.project_name
  }
}

resource "aws_s3_bucket_public_access_block" "assets" {
  bucket = aws_s3_bucket.assets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Upload every file in training_data/ (same folder the rest of this repo's
# training scripts already read from) under s3://<bucket>/training_data/.
resource "aws_s3_object" "training_data" {
  for_each = fileset(var.training_data_dir, "**")

  bucket = aws_s3_bucket.assets.id
  key    = "training_data/${each.value}"
  source = "${var.training_data_dir}/${each.value}"
  etag   = filemd5("${var.training_data_dir}/${each.value}")
}

resource "aws_s3_object" "finetune_script" {
  bucket = aws_s3_bucket.assets.id
  key    = "scripts/finetune_llama70b_qlora.py"
  source = "${path.module}/scripts/finetune_llama70b_qlora.py"
  etag   = filemd5("${path.module}/scripts/finetune_llama70b_qlora.py")
}
