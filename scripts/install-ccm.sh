#!/bin/bash
# Claude Code Profile 安装脚本 (API Token 认证)
# 用法: bash install-ccm.sh

set -e

PROFILE='{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "alwaysThinkingEnabled": true,
  "enabledPlugins": {
    "figma@claude-plugins-official": true,
    "frontend-design@claude-plugins-official": true,
    "github@claude-plugins-official": true,
    "superpowers@claude-plugins-official": true,
    "swift-lsp@claude-plugins-official": true
  },
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "sk-cp--1cmU94FyZkyEnDGaMBUCpXj5j3FZv794asxaK1WYWPuT_Px1Z1Jcf9ddImoRcBhgsOWR2rXXaMGwFCz3cT0TKqsHG29k5XzckUEV9XvQrE6nUvJgUFWeow",
    "ANTHROPIC_BASE_URL": "https://api.minimaxi.com/anthropic",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMax-M2.7-highspeed",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "MiniMax-M2.7-highspeed",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M2.7-highspeed",
    "ANTHROPIC_MODEL": "MiniMax-M2.7-highspeed",
    "ANTHROPIC_REASONING_MODEL": "MiniMax-M2.7-highspeed",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1
  },
  "feedbackSurveyState": {
    "lastShownTime": 1755760737104
  },
  "permissions": {
    "allow": [
      "Bash(curl:*)",
      "Bash(ls:*)",
      "mcp__pencil"
    ]
  }
}'

echo "==> 1. 安装 Claude Code (npm)"
npm install -g @anthropic-ai/claude-code

echo "==> 2. 创建 ~/.claude 目录"
mkdir -p ~/.claude

echo "==> 3. 写入 settings.json"
echo "$PROFILE" > ~/.claude/settings.json

echo "==> 4. 写入环境变量到 ~/.zshrc (或 ~/.bashrc)"
SHELL_RC="$HOME/.zshrc"
[ -n "$BASH_VERSION" ] && SHELL_RC="$HOME/.bashrc"

ENV_BLOCK='
# Claude Code profile
export ANTHROPIC_AUTH_TOKEN="sk-cp--1cmU94FyZkyEnDGaMBUCpXj5j3FZv794asxaK1WYWPuT_Px1Z1Jcf9ddImoRcBhgsOWR2rXXaMGwFCz3cT0TKqsHG29k5XzckUEV9XvQrE6nUvJgUFWeow"
export ANTHROPIC_BASE_URL="https://api.minimaxi.com/anthropic"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="MiniMax-M2.7-highspeed"
export ANTHROPIC_DEFAULT_OPUS_MODEL="MiniMax-M2.7-highspeed"
export ANTHROPIC_DEFAULT_SONNET_MODEL="MiniMax-M2.7-highspeed"
export ANTHROPIC_MODEL="MiniMax-M2.7-highspeed"
export ANTHROPIC_REASONING_MODEL="MiniMax-M2.7-highspeed"
export API_TIMEOUT_MS="3000000"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
'

# 检查是否已存在，避免重复写入
if ! grep -q "ANTHROPIC_AUTH_TOKEN" "$SHELL_RC" 2>/dev/null; then
    echo "$ENV_BLOCK" >> "$SHELL_RC"
    echo "    已追加环境变量到 $SHELL_RC"
else
    echo "    环境变量已存在，跳过"
fi

echo ""
echo "==> 安装完成！"
echo ""
echo "    重新加载 shell 配置: source $SHELL_RC"
echo "    启动:                  claude"
echo ""
echo "    settings.json 已配置在 ~/.claude/settings.json"
echo "    插件 (figma, frontend-design, github, superpowers, swift-lsp) 会在首次启动时自动安装"
