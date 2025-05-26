# SmarTTS

本项目利用 NVIDIA AIQ Toolkit 构建一个智能系统，能够将输入的文本文章转换为包含多角色、富有情感的语音输出，旨在提供接近有声剧的听觉体验。系统通过多个基于 Agno 框架的智能体（Agent）协作完成，包括文本分割、角色语音选择和语音合成等步骤，并通过自定义的 MCP (Model Context Protocol) 服务器调用 Minimax TTS 服务进行实际的语音生成。

## 主要功能

* **文本智能分割**: 使用大语言模型（LLM）分析文章结构，自动识别旁白和不同角色的对话内容。
* **角色语音自动选择**: 通过 MCP 工具调用 Minimax 服务获取可用声音列表，并为文章中的不同角色智能分配声音。
* **分段语音合成**: 为每个分割后的文本片段，使用分配的声音通过 Minimax TTS 服务生成独立的音频片段（通常以URL形式返回）。
* **音频下载与合并**: 自动下载生成的音频片段，并使用 Pydub 库按照原文顺序将它们无缝合并成一个完整的音频文件。
* **多智能体协作**: 基于 Agno 框架构建多个智能体，分别负责文本处理、语音选择和音频生成等任务，协同完成整个工作流程。
* **AIQ Toolkit 集成**: 整个工作流在 NVIDIA AIQ Toolkit 框架下运行，LLM 等核心组件由 AIQ Toolkit 管理和提供。

## 技术栈

* **核心框架**: NVIDIA AIQ Toolkit
* **智能体构建**: Agno Framework
* **大语言模型 (LLM)**: 通过 AIQ Toolkit 配置和加载 (例如 NVIDIA NIM LLM `meta/llama-4-maverick-17b-128e-instruct`，或配置为 OpenAI 兼容接口的 DeepSeek 等模型)
* **语音合成 (TTS)**: Minimax TTS 服务 (通过用户自定义的 MCP 服务器桥接)
* **MCP 服务器**: 用户本地搭建的，用于 Agno Agent 与 Minimax TTS 服务进行通信。本项目依赖此服务器的正确运行。
* **音频处理**: Pydub
* **编程语言**: Python 3.x (大量使用 `asyncio` 进行异步处理)
* **主要依赖库**: `agno`, `pydub`, `httpx`, `aiofiles`, `aiq-toolkit` (或相关AIQ包)

## 环境准备与依赖安装

1.  **Python 环境**: 推荐 Python 3.10 或更高版本。
2.  **NVIDIA AIQ Toolkit**: 确保已正确安装并配置了 NVIDIA AIQ Toolkit。请遵循官方文档进行安装。
    然后通过 `pip install -r requirements.txt` 安装。
3.  **FFmpeg**: Pydub 库进行音频格式处理（如导出为MP3）通常需要 FFmpeg。请确保 FFmpeg 已安装在你的系统上，并且其可执行文件路径已添加到系统环境变量 `PATH` 中。
4.  **本地 MCP 服务器**:
    * 用户需要自行搭建并运行一个 MCP 服务器。此服务器应能接收来自 `Agno MCPTools` 的请求，并将其转发给实际的 Minimax TTS 服务接口。
    * 确保此 MCP 服务器暴露了名为 `list_voices` (无参数) 和 `text_to_audio` (参数: `text`, `voice_id`, `outputFile` - 其中 `outputFile` 可为占位符，实际返回音频URL) 的工具。
    * 在 `config.yml` 中配置正确的 MCP 服务器 URL。
5.  **Minimax TTS 服务**:
    * 确保你的 MCP 服务器能够成功调用 Minimax TTS 服务，并且你拥有有效的 Minimax 服务凭证（如果需要）。

## 项目配置 (`config.yml`)

项目通过 `config.yml` 文件进行配置。以下是一个配置示例和关键项说明：

```yaml
# config.yml

general:
  use_uvloop: true # 可选，通常用于提升asyncio性能
  logging:
    version: 1
    disable_existing_loggers: false
    formatters:
      simple:
        format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        level: DEBUG # 建议设为 DEBUG 以获取详细日志，包括Agno工具调用
        stream: ext://sys.stdout
    root:
      handlers: [console]
      level: DEBUG
    loggers:
      agno: 
        level: DEBUG # Agno 库的详细日志
      # 你的Python文件名 (不含.py) 将是logger名，例如 aiq_tts_workflow_from_test
      aiq_tts_workflow_from_test: # 确保与 getLogger(__name__) 在你的主Python文件中生成的名称匹配
        level: DEBUG 
      httpx:
        level: INFO

llms:
  # 配置你希望AIQ Toolkit使用的LLM
  # 示例：使用NIM LLM (你需要确保AIQ环境已配置好NIM)
  nim_llm_for_tts: 
    _type: nim
    model_name: meta/llama-4-maverick-17b-128e-instruct # 替换为你想用的NIM模型
    temperature: 0.0
    # ... 其他NIM特定参数，如果需要 ...

  # 示例：或者配置一个OpenAI兼容的LLM (如本地部署的或DeepSeek的兼容API)
  # deepseek_compatible_llm:
  #   _type: openai_compatible
  #   model_name: "deepseek-chat"
  #   base_url: "YOUR_OPENAI_COMPATIBLE_ENDPOINT_URL" # 例如 http://localhost:8000/v1
  #   api_key: "YOUR_API_KEY_IF_ANY" 
  #   temperature: 0.1

functions: {} # 对于单个主工作流，如果其配置直接在下面的 `workflow:` 块中定义，此部分可省略或为空

workflow:
  _type: aiq_custom_tts_from_test # **重要**: 必须匹配Python代码中AIQCustomTTSConfig的`name`
  llm_name: nim_llm_for_tts      # **重要**: 引用上面`llms:`部分定义的LLM配置名

  temp_audio_directory_path: "./aiq_generated_audio" # 临时和最终音频文件的存放路径
  mcp_server_url: "http://localhost:3000/sse" # **重要**: 你的本地MCP服务器URL
                                              # 或者 ModelScope 公开的示例URL (如果适用且有效)
  mcp_client_timeout: 60               # MCP客户端连接和请求的超时时间 (秒)
  mcp_client_sse_read_timeout: 60      # MCP客户端SSE流读取超时 (秒)
  mcp_tools_overall_timeout: 180       # MCPTools上下文管理器的整体超时 (秒)
