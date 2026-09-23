# Holds the Hugging Face token as a SecureString so it never appears in
# plaintext user_data or instance metadata. The instance role below is
# scoped to read *only* this one parameter.
resource "aws_ssm_parameter" "hf_token" {
  name        = "/${var.project_name}/hf_token"
  description = "Hugging Face access token used to download the gated Llama checkpoint."
  type        = "SecureString"
  value       = var.hf_token

  tags = {
    Project = var.project_name
  }
}
