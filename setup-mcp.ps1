$env:EMBEDDING_PROVIDER="OpenAI"
$env:EMBEDDING_MODEL="Qwen/Qwen3-VL-Embedding-8B"
$env:EMBEDDING_BATCH_SIZE="75"
$env:OPENAI_API_KEY="sk-dsgdtybpzhaaojtigvpwdsxhfiwryqlulveunkhwuryxbbby"
$env:OPENAI_BASE_URL="https://api.siliconflow.cn/v1/embeddings"
$env:MILVUS_ADDRESS="https://in03-e79eedc9b582341.serverless.aws-eu-central-1.cloud.zilliz.com"
$env:MILVUS_TOKEN="a5f72ba61e56940d3935a37db64354c483fd3b36701dabbb7b10c063e01373472deddd81f15a5512ecc90941d8ff10fd263e6ab0"
$env:SPLITTER_TYPE="ast"

claude mcp add claude-context 
  -e OPENAI_API_KEY="$env:OPENAI_API_KEY" `
  -e OPENAI_BASE_URL="$env:OPENAI_BASE_URL" `
  -e EMBEDDING_MODEL="$env:EMBEDDING_MODEL" `
  -e EMBEDDING_PROVIDER="$env:EMBEDDING_PROVIDER" `
  -e MILVUS_ADDRESS="$env:MILVUS_ADDRESS" `
  -e MILVUS_TOKEN="$env:MILVUS_TOKEN" `
  -e SPLITTER_TYPE="$env:SPLITTER_TYPE" `
  -- npx @zilliz/claude-context-mcp@latest
