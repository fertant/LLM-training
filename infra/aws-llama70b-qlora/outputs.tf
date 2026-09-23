output "instance_id" {
  value = aws_instance.finetune.id
}

output "instance_public_ip" {
  value = aws_instance.finetune.public_ip
}

output "s3_bucket" {
  description = "Holds the uploaded training_data/, the fine-tuning script, and (once done) output/ with the LoRA adapter."
  value       = aws_s3_bucket.assets.bucket
}

output "tail_logs_command" {
  description = "Run once you can reach the instance, to watch bootstrap + training progress."
  value       = "aws ssm start-session --target ${aws_instance.finetune.id} --region ${var.aws_region} --document-name AWS-StartInteractiveCommand --parameters command=\"tail -f /var/log/finetune.log\""
}

output "download_adapter_command" {
  description = "Run after training finishes to pull the fine-tuned LoRA adapter down locally."
  value       = "aws s3 sync s3://${aws_s3_bucket.assets.bucket}/output ./output"
}
