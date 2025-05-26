# aiq_tts_workflow_from_test.py

import asyncio
import logging
import os
import pathlib
import json
import re
from textwrap import dedent
from typing import List, Dict, Any, Tuple
import uuid

# AIQ Toolkit Imports
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

# Original Imports from your successful test.py
from agno.agent import Agent
# from agno.models.deepseek import DeepSeek # 将被 AIQ LLM 替代
from agno.tools.mcp import MCPTools, SSEClientParams
from pydub import AudioSegment
import httpx 
import aiofiles
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__) # AIQ 会处理全局日志配置

# --- AIQ 配置类 ---
class AIQCustomTTSConfig(FunctionBaseConfig, name="aiq_custom_tts_from_test"):
    llm_name: LLMRef # 用于指定在 config.yml 中定义的 LLM 配置
    temp_audio_directory_path: str = "./aiq_toolkitted_temp_audio" # 可配置的临时音频目录
    mcp_server_url: str = "http://localhost:3000/sse" # 默认 ModelScope URL
    mcp_client_timeout: int = 45 
    mcp_client_sse_read_timeout: int = 45 
    mcp_tools_overall_timeout: int = 120

# --- 从你成功的 test.py 复制的 download_audio 函数 (保持原样) ---
async def download_audio(url: str, output_dir: pathlib.Path, file_index: int = 0) -> Tuple[bool, pathlib.Path]:
    """
    从 URL 异步下载音频文件并将其保存到 output_dir。
    文件名将是唯一的，包含 file_index 和一个 UUID。
    在确定文件扩展名之前，会剥离 URL 中的查询参数。
    返回一个元组：(success_boolean, path_to_saved_or_attempted_file)。
    """
    output_path = output_dir / f"download_error_{file_index}_{uuid.uuid4().hex[:8]}.error" # 默认错误输出路径

    try:
        parsed_url = urlparse(url)
        url_path_part = parsed_url.path
        unquoted_url_path = unquote(url_path_part) # 解码URL路径部分
        file_extension = pathlib.Path(unquoted_url_path).suffix # 获取文件扩展名

        if not file_extension: # 如果从路径中未获取到扩展名
            match_ext = re.search(r'\.(mp3|wav|ogg|aac|m4a|flac)(\?|$)', url, re.IGNORECASE) # 尝试从完整URL中正则匹配
            if match_ext:
                file_extension = "." + match_ext.group(1)
            else:
                file_extension = ".mp3" # 默认为 .mp3
        
        if file_extension and not file_extension.startswith('.'): # 确保扩展名以点开头
            file_extension = '.' + file_extension
        elif not file_extension: # 如果仍然没有扩展名
            file_extension = ".mp3" # 默认为 .mp3

        safe_filename = f"downloaded_segment_{file_index}_{uuid.uuid4().hex[:8]}{file_extension}" # 生成安全的文件名
        output_path = output_dir / safe_filename # 最终输出路径

        async with httpx.AsyncClient(timeout=30.0) as client: # 使用 httpx 异步下载 (超时保持和test.py一致)
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status() # 如果状态码不是 2xx，则抛出异常
            async with aiofiles.open(output_path, 'wb') as f: # 异步写入文件
                await f.write(response.content)
        logger.info(f"从 {url} 下载音频到 {output_path}")
        return True, output_path
    except httpx.HTTPStatusError as e:
        logger.error(f"下载 {url} 到 {output_path} 时发生 HTTP 错误: {e}. 响应: {e.response.text if e.response else '无响应'}")
        return False, output_path
    except httpx.RequestError as e:
        logger.error(f"下载 {url} 到 {output_path} 时发生请求错误: {e}")
        return False, output_path
    except Exception as e:
        logger.error(f"下载 {url} 到 {output_path} 时发生意外错误: {e}", exc_info=True)
        return False, output_path

# --- AIQ 注册函数 ---
@register_function(config_type=AIQCustomTTSConfig, framework_wrappers=[LLMFrameworkEnum.AGNO])
async def aiq_tts_entry_point(config: AIQCustomTTSConfig, builder: Builder):
    """
    AIQ Toolkit 的入口函数，用于文章转语音工作流。
    """
    # 1. 从 AIQ Builder 获取 LLM 实例
    # 假设 config.llm_name 指向一个与 Agno.Agent 兼容的 LLM 配置
    # (例如，一个 OpenAI 兼容的模型，或者 AIQ 为 DeepSeek 提供的 Agno 包装器)
    llm_instance_from_aiq = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.AGNO)
    logger.info(f"LLM 实例 '{config.llm_name}' 已从 AIQ Builder 获取。")

    # 2. 从配置设置临时音频目录
    # 注意：test.py 中的 temp_audio_directory 是全局定义的。
    # 这里我们从 AIQ config 获取，并确保它在 _process_article 中可用。
    # 全局的 temp_audio_directory 定义（如果存在于此文件顶部）将被此处的配置值覆盖（在_process_article内）。
    # 为了与 test.py 结构保持一致，我们将实际的 temp_audio_directory 路径传递给 _process_article。
    current_temp_audio_dir = pathlib.Path(config.temp_audio_directory_path)
    os.makedirs(current_temp_audio_dir, exist_ok=True)
    logger.info(f"临时音频目录设置为: {current_temp_audio_dir}")

    # 3. 定义 Agent (在 AIQ 函数内部，使用 AIQ 提供的 LLM)
    # text_split_agent 的定义与 test.py 完全一致，除了 model 参数
    text_split_agent_for_aiq = Agent(
        name="text_split_agent", # 与 test.py 中的 name 一致
        role="Text splitter for article content", # 与 test.py 中的 role 一致
        model=llm_instance_from_aiq, # 使用 AIQ 提供的 LLM
        description=dedent("""
        Analyzes input text and splits it into segments by character or identifies it as a single-speaker document.
        """), # 与 test.py 中的 description 一致
        instructions=[ # 与 test.py 中的 instructions 完全一致
            "You will be given an article text.",
            "Your primary task is to analyze the text and determine if it contains multiple speaking characters and dialogue, or if it is a single-speaker document (e.g., a news report, a speech, a technical article, a first-person narrative without dialogue with others).",
            "Split the text into segments based on characters and narration.",
            "Create a JSON list of objects, where each object has 'character_id' (string) and 'text_segment' (string).",
            "If multiple consecutive text segments belong to the 'Narrator', merge them into a single 'text_segment' for the 'Narrator'.",
            "Attribute ONLY the exact text enclosed in quotation marks to a speaking character. All other text (descriptions, actions, etc.) MUST be attributed to the 'Narrator'.",
            "If the text is from a single speaker or has no dialogue, output a JSON list containing a single object with 'character_id' (e.g., 'Narrator') and the ENTIRE original text as 'text_segment'.",
            "Your output MUST be a valid JSON list.",
            "**CRITICALLY IMPORTANT: Your entire response must be ONLY the valid JSON list itself, starting with '[' and ending with ']'. Do NOT include any other text, explanations, conversation, headings, comments, or markdown code block formatting (like ```json) surrounding the JSON data. Provide ONLY the raw JSON content.**",
            "Now, process the provided article text based on these instructions, first determining if it's a multi-dialogue scenario or a single-speaker/no-dialogue scenario."
        ],
        show_tool_calls=True # 保持与你成功脚本中可能有的调试设置一致或添加以进行调试
    )

    # 核心工作流逻辑，封装在内部异步函数中，以便 AIQ 调用
    async def _process_article(article_file_path: str) -> str: # AIQ 通常期望返回字符串
        """
        包含主要工作流逻辑的内部函数。
        此函数的内容几乎完全复制自你的 test.py 中的 article_to_speech_workflow。
        """
        paths_generated_by_download_audio: List[pathlib.Path] = []
        # final_audio_path_str: str | None = None # 在 test.py 中未使用，这里也注释掉

        try:
            logger.info(f"从以下路径读取文章 (AIQ): {article_file_path}")
            article_text = pathlib.Path(article_file_path).read_text(encoding="utf-8")
            
            if not article_text.strip():
                logger.info("输入文章为空 (AIQ)。")
                return json.dumps({"status": "success", "details": "输入文章为空。"})

            logger.info("将文本分割成片段 (AIQ)...")
            # 使用在 aiq_tts_entry_point 中定义的 text_split_agent_for_aiq
            text_split_response = await text_split_agent_for_aiq.arun(article_text)
            logger.info(f"文本分割代理原始响应 (AIQ): {text_split_response.content}")
            
            # !!! 关键修正：清理可能由LLM（在AIQ环境中）添加的Markdown代码块标记 !!!
            content_str = text_split_response.content
            cleaned_content_str = re.sub(r"^\s*```json\s*|^\s*```\s*|\s*```\s*$", "", content_str, flags=re.MULTILINE).strip()

            try:
                segmented_texts = json.loads(cleaned_content_str) # 使用清理后的字符串
                if not isinstance(segmented_texts, list):
                    logger.error(f"文本分割代理未返回 JSON 列表 (AIQ): {segmented_texts}")
                    return json.dumps({"status": "failure", "details": "文本分割代理未返回 JSON 列表。"})
            except json.JSONDecodeError as e:
                logger.error(f"从 text_split_agent 解码 JSON 失败 (AIQ): {e}. 清理后的响应为: '{cleaned_content_str}'")
                return json.dumps({"status": "failure", "details": f"从 text_split_agent 解码 JSON 失败: {e}"})

            indexed_segments = [{"index": i, **segment} for i, segment in enumerate(segmented_texts)]

            server_params = SSEClientParams(
                url=config.mcp_server_url, # 从 AIQ config 获取
                timeout=config.mcp_client_timeout, # 从 AIQ config 获取
                sse_read_timeout=config.mcp_client_sse_read_timeout # 从 AIQ config 获取
            )
            
            segments_with_voices_data: List[Dict[str, Any]] = []
            tts_results: Dict[str, Any] = {} # 与 test.py 保持一致

            # MCPTools 上下文管理器，使用 AIQ config 中的超时设置
            async with MCPTools(server_params=server_params, timeout_seconds=config.mcp_tools_overall_timeout, transport="sse") as mcp_tools_obj:
                # Agent 1 (在 test.py 中是 Agent 2): Voice Selector Agent
                # 定义与 test.py 完全一致，除了 model 和 tools 参数
                voice_selector_agent = Agent(
                    name="Voice_Selector_Agent", # 与 test.py 中的 name 一致
                    role="Selects appropriate voices for text segments using Minimax service.", # 与 test.py 中的 role 一致
                    model=llm_instance_from_aiq, # 使用 AIQ 提供的 LLM
                    description=dedent("""
                    Takes a list of text segments, fetches available voices using a tool,
                    and assigns a voice_id to each segment based on character_id and voice availability.
                    Outputs a list of segments augmented with voice_id.
                    """), # 与 test.py 中的 description 一致
                    tools=[mcp_tools_obj], # 使用已初始化的 MCPTools 实例
                    instructions=[ # 与 test.py 中的 instructions 完全一致
                        "You will be given a JSON list of text segments. Each item has 'index', 'character_id', and 'text_segment'.",
                        "Your first step is to call the `list_voices` tool to get a list of available voice objects. Each voice object typically contains 'voice_id' and 'name'.",
                        "If the `list_voices` tool call fails or returns an empty list of voices, you MUST respond with a JSON object: {\"error\": \"Failed to retrieve voices or no voices available.\"}. Do NOT proceed further.",
                        "If voices are available, your goal is to assign a 'voice_id' to each input text segment.",
                        "Maintain a mapping for character_id to voice_id to ensure consistency: if a character_id appears multiple times, it should be assigned the same voice_id.",
                        "Assign voices as follows:",
                        "  - For 'Narrator' character_id: Pick the first voice from the `list_voices` result. If you can identify a voice more suitable for narration (e.g., based on name like 'Standard', 'Narration', 'MaleA', 'FemaleA'), prefer that. Ensure this choice is consistent for all 'Narrator' segments.",
                        "  - For other character_ids: Assign voices from the available pool. Try to give different characters different voices. If there are more unique character_ids (excluding Narrator) than remaining voices, cycle through the available voices for these other characters.",
                        "  - If only one voice is available from `list_voices`, use that single voice_id for all segments.",
                        "Iterate through the input segments and add a 'voice_id' field to each segment object based on your selection logic.",
                        "Your final output MUST be a valid JSON list of these augmented segment objects: [{'index': ..., 'character_id': ..., 'text_segment': ..., 'voice_id': ...}, ...].",
                        "**CRITICALLY IMPORTANT: If `list_voices` fails or no voices are available, your entire response must be ONLY the JSON error object specified above. Otherwise, your entire response must be ONLY the valid JSON list of segments with assigned voice_ids. Do NOT include any other text, explanations, conversation, headings, comments, or markdown code block formatting (like ```json) surrounding the JSON data.**"
                    ],
                    show_tool_calls=True # 保持或添加以进行调试
                )
                
                logger.info("为片段选择语音 (AIQ)...")
                voice_selection_response = await voice_selector_agent.arun(json.dumps(indexed_segments))
                logger.info(f"语音选择器代理原始响应 (AIQ): {voice_selection_response.content}")
                
                # !!! 关键修正：清理可能由LLM（在AIQ环境中）添加的Markdown代码块标记 !!!
                vs_content_str = voice_selection_response.content
                vs_cleaned_content_str = re.sub(r"^\s*```json\s*|^\s*```\s*|\s*```\s*$", "", vs_content_str, flags=re.MULTILINE).strip()

                try:
                    parsed_voice_selection = json.loads(vs_cleaned_content_str) # 使用清理后的字符串
                    if isinstance(parsed_voice_selection, dict) and "error" in parsed_voice_selection:
                        logger.error(f"语音选择失败 (AIQ): {parsed_voice_selection['error']}")
                        return json.dumps({"status": "failure", "details": f"语音选择失败: {parsed_voice_selection['error']}"})
                    
                    is_valid_list = isinstance(parsed_voice_selection, list)
                    if is_valid_list and parsed_voice_selection:
                        if not all(isinstance(s, dict) and 'voice_id' in s and 'index' in s and 'character_id' in s and 'text_segment' in s for s in parsed_voice_selection):
                            is_valid_list = False

                    if not is_valid_list:
                        logger.error(f"语音选择器代理未返回包含所需字段的有效片段列表 (AIQ): {parsed_voice_selection}")
                        return json.dumps({"status": "failure", "details": "语音选择未返回预期的列表结构或内容。"})
                    segments_with_voices_data = parsed_voice_selection
                except json.JSONDecodeError as e:
                    logger.error(f"从语音选择器代理解码 JSON 失败 (AIQ): {e}. 清理后的响应为: '{vs_cleaned_content_str}'")
                    return json.dumps({"status": "failure", "details": f"从语音选择器代理解码 JSON 失败: {e}"})

                # Agent 2 (在 test.py 中是 Agent 3): Audio Producer Agent
                # 定义与 test.py 完全一致，除了 model 和 tools 参数
                audio_producer_agent = Agent(
                    name="Audio_Producer_Agent", # 与 test.py 中的 name 一致
                    role="Generates audio for text segments using pre-selected voices via Minimax service.", # 与 test.py 中的 role 一致
                    model=llm_instance_from_aiq, # 使用 AIQ 提供的 LLM
                    description=dedent(f"""
                    Takes a list of text segments, each with a pre-assigned 'voice_id'.
                    Calls a TTS tool for each segment to generate an audio URL.
                    Outputs a JSON object containing lists of successful audio URLs and any failed segments.
                    """), # 与 test.py 中的 description 一致 (temp_audio_directory 的引用方式稍有不同，因为它是函数参数)
                    tools=[mcp_tools_obj], # 使用已初始化的 MCPTools 实例
                    instructions=[ # 与 test.py 中的 instructions 完全一致
                        "You will be given a JSON list of text segments. Each item has 'index', 'character_id', 'text_segment', and 'voice_id'. The 'voice_id' has been pre-selected.",
                        "Your task is to generate audio for each segment using the provided 'voice_id' and 'text_segment' by calling the `text_to_audio` tool.",
                        "Iterate through the input list of segments. For each segment:",
                        "  - If the 'text_segment' is empty or consists only of whitespace, record this segment as a failure with reason 'Text is empty'. Do not call the tool.",
                        "  - If the 'voice_id' is missing, empty, or indicates a known issue (e.g., a placeholder like 'NO_VOICE_AVAILABLE' or 'VOICE_SELECTION_FAILED'), record this segment as a failure with reason 'Voice ID missing or invalid'. Do not call the tool.",
                        "  - Otherwise, call the `text_to_audio` tool with the 'text_segment', the 'voice_id', and a placeholder `outputFile` path (e.g., 'temp_audio/segment.mp3'). Capture the URL returned by the `text_to_audio` tool if successful.",
                        "  - If the `text_to_audio` tool call is successful, record the returned audio URL. Include the original segment's 'index'.",
                        "  - If the `text_to_audio` tool call fails (e.g., due to rate limits, service errors), record this segment as a failure. Include the segment's original 'index', 'character_id', 'text_segment', and a 'reason' (e.g., 'Minimax TTS tool failed: <error_details from tool>').",
                        "After processing all segments, your *sole and absolute final response* MUST be a single, valid JSON object with exactly two keys:",
                        "1.  `generated_audio_urls`: A JSON list of objects. Each object should have 'index' (original segment index) and 'url' (the audio URL).",
                        "2.  `failed_segments_info`: A JSON list of objects. Each object contains 'index' (integer), 'character_id' (string), 'text_segment' (string), and 'reason' (string).",
                        "If there are no successful conversions or no failures, the corresponding list should be empty (`[]`).",
                        "**CRITICALLY IMPORTANT: Your entire final response must ONLY be this JSON object. Do NOT include any other text, explanations, conversation, headings, comments, or markdown code block formatting (like ```json) surrounding the JSON data. Provide ONLY the raw JSON object content.**"
                    ],
                    show_tool_calls=True # 保持或添加以进行调试
                )

                logger.info("为选定语音的片段生成音频 (AIQ)...")
                audio_producer_input = json.dumps(segments_with_voices_data if segments_with_voices_data else [])
                tts_response_obj = await audio_producer_agent.arun(audio_producer_input)
                logger.info(f"音频生成器代理原始响应 (AIQ): {tts_response_obj.content}")

                # !!! 关键修正：清理可能由LLM（在AIQ环境中）添加的Markdown代码块标记 !!!
                ap_content_str = tts_response_obj.content
                ap_cleaned_content_str = re.sub(r"^\s*```json\s*|^\s*```\s*|\s*```\s*$", "", ap_content_str, flags=re.MULTILINE).strip()
                
                try:
                    tts_results = json.loads(ap_cleaned_content_str) # 使用清理后的字符串
                    if not isinstance(tts_results, dict) or \
                       "generated_audio_urls" not in tts_results or not isinstance(tts_results["generated_audio_urls"], list) or \
                       "failed_segments_info" not in tts_results or not isinstance(tts_results["failed_segments_info"], list):
                        logger.error(f"从音频生成器代理输出解析的 JSON 没有预期的结构 (AIQ): {tts_results}")
                        return json.dumps({"status": "failure", "details": "音频生成器代理返回了 JSON，但结构意外。"})
                except json.JSONDecodeError as e:
                    logger.error(f"从音频生成器代理解码 JSON 失败 (AIQ): {e}. 清理后的响应为: '{ap_cleaned_content_str}'")
                    return json.dumps({"status": "failure", "details": f"从音频生成器代理解码 JSON 失败: {e}"})
            
            # MCPTools context ends here

            # --- 后续逻辑与 test.py 中的 article_to_speech_workflow 完全一致 ---
            generated_url_objects = tts_results.get("generated_audio_urls", [])
            failed_segments_info = tts_results.get("failed_segments_info", [])

            if failed_segments_info:
                logger.warning(f"音频生成器代理报告了 {len(failed_segments_info)} 个失败的片段 (AIQ):")
                for failure in failed_segments_info:
                    logger.warning(f"  索引: {failure.get('index', 'N/A')}, 角色: {failure.get('character_id', 'N/A')}, 原因: {failure.get('reason', 'Unknown')}")

            if not generated_url_objects:
                status_detail = "音频生成器代理未成功生成任何音频 URL。"
                if failed_segments_info:
                    status_detail += f" 所有片段 TTS 失败 ({len(failed_segments_info)} 个失败)。"
                if not indexed_segments and not segments_with_voices_data: 
                    return json.dumps({"status": "success", "details": "输入文章为空或导致没有文本片段需要进行 TTS 处理。"})
                return json.dumps({
                    "status": "failure" if failed_segments_info or segments_with_voices_data else "success", 
                    "details": status_detail,
                    "failed_segments": failed_segments_info
                })

            logger.info(f"尝试下载 {len(generated_url_objects)} 个音频文件 (AIQ)...")
            download_tasks = []
            for url_obj in generated_url_objects:
                url_str = url_obj.get("url")
                original_idx = url_obj.get("index", -1)
                if not url_str or original_idx == -1:
                    logger.warning(f"来自音频生成器代理的无效 URL 对象 (AIQ): {url_obj}")
                    continue
                # download_audio 现在只需要 url, output_dir, file_index
                download_tasks.append(download_audio(url_str, current_temp_audio_dir, original_idx))


            download_results_list = await asyncio.gather(*download_tasks, return_exceptions=True)
            successfully_downloaded_map: Dict[int, pathlib.Path] = {}

            for i, dl_result in enumerate(download_results_list):
                if isinstance(dl_result, BaseException):
                    failed_url_obj = generated_url_objects[i] if i < len(generated_url_objects) else {"url": "未知 URL", "index": "未知索引"}
                    logger.error(f"URL '{failed_url_obj.get('url')}' (原始片段索引: {failed_url_obj.get('index')}) 的下载任务因异常失败 (AIQ): {dl_result}")
                    continue

                success, file_path_obj = dl_result
                if success and file_path_obj and file_path_obj.exists():
                    paths_generated_by_download_audio.append(file_path_obj)
                    match = re.search(r"downloaded_segment_(\d+)_", file_path_obj.name)
                    if match:
                        original_idx_from_name = int(match.group(1))
                        successfully_downloaded_map[original_idx_from_name] = file_path_obj
                    else:
                        logger.warning(f"无法从文件名中提取原始索引 (AIQ): {file_path_obj.name}")
                elif success and (not file_path_obj or not file_path_obj.exists()):
                     logger.warning(f"下载任务 (任务索引 {i}) 报告成功，但文件 {file_path_obj} 无效或不存在 (AIQ)。")
                elif not success:
                     logger.warning(f"下载任务 (任务索引 {i}) 如 download_audio 报告失败。尝试的路径: {file_path_obj} (AIQ)")
            
            if not successfully_downloaded_map:
                status_detail = f"音频生成器生成了 {len(generated_url_objects)} 个 URL，但未能成功下载任何一个。"
                if failed_segments_info:
                    status_detail += f" 此外，在 TTS 过程中有 {len(failed_segments_info)} 个片段失败。"
                return json.dumps({
                    "status": "failure",
                    "details": status_detail,
                    "failed_segments": failed_segments_info
                })
            
            logger.info(f"成功下载 {len(successfully_downloaded_map)} 个音频文件。按顺序合并它们 (AIQ)...")
            combined_audio = AudioSegment.empty()
            sorted_indices = sorted(successfully_downloaded_map.keys())

            for original_idx in sorted_indices:
                audio_file_path = successfully_downloaded_map[original_idx]
                try:
                    file_format = audio_file_path.suffix[1:].lower() if audio_file_path.suffix else "mp3"
                    if file_format not in ["mp3", "wav", "ogg", "flv", "aac", "mpeg", "m4a"]:
                        logger.warning(f"文件 {audio_file_path} 中 Pydub 不支持的音频格式 '{file_format}'。尝试作为 mp3 加载。 (AIQ)")
                        file_format = "mp3"
                    segment_audio = AudioSegment.from_file(audio_file_path, format=file_format)
                    combined_audio += segment_audio
                except Exception as e_load:
                    logger.error(f"加载已下载的音频文件 {audio_file_path} 以进行串联时出错 (AIQ): {e_load}")

            if len(combined_audio) > 0:
                run_specific_id = str(uuid.uuid4())[:8]
                final_filename = f"final_article_speech_{run_specific_id}.mp3"
                final_audio_full_path = current_temp_audio_dir / final_filename # 使用从config来的路径
                final_audio_path_str_for_return = str(final_audio_full_path) # 之前这里变量名是 final_audio_path_str

                try:
                    combined_audio.export(final_audio_full_path, format="mp3")
                    logger.info(f"最终组合音频已保存到 (AIQ): {final_audio_full_path}")
                    details_msg = f"音频生成成功。{len(successfully_downloaded_map)}/{len(generated_url_objects)} 个 URL 已下载并合并。"
                    if failed_segments_info:
                        details_msg += f" 注意: {len(failed_segments_info)} 个片段在 TTS 过程中失败。"
                    return json.dumps({
                        "status": "success",
                        "final_audio_path": final_audio_path_str_for_return,
                        "details": details_msg,
                        "failed_segments": failed_segments_info
                    })
                except Exception as e_export:
                    logger.error(f"将组合音频导出到 {final_audio_full_path} 失败 (AIQ): {e_export}")
                    return json.dumps({"status": "failure", "details": f"保存最终音频失败: {e_export}", "failed_segments": failed_segments_info})
            else:
                status_detail = f"音频生成器生成了 {len(generated_url_objects)} 个 URL，并下载了 {len(successfully_downloaded_map)} 个，但无法加载/合并到最终音频中。"
                if failed_segments_info:
                    status_detail += f" 此外，在 TTS 过程中有 {len(failed_segments_info)} 个片段失败。"
                return json.dumps({
                    "status": "failure",
                    "details": status_detail,
                    "failed_segments": failed_segments_info
                })

        except Exception as e_main:
            logger.exception("article_to_speech_workflow (_process_article for AIQ) 中发生未处理的错误")
            current_failed_segments = tts_results.get("failed_segments_info") if 'tts_results' in locals() and isinstance(tts_results, dict) else []
            return json.dumps({
                "status": "failure",
                "details": f"工作流程错误 (AIQ): {str(e_main)}",
                "failed_segments": current_failed_segments
            })
        finally:
            logger.info(f"清理 {len(paths_generated_by_download_audio)} 个临时音频片段文件 (AIQ)...")
            for temp_file in paths_generated_by_download_audio:
                if temp_file and isinstance(temp_file, pathlib.Path) and temp_file.exists():
                    try:
                        os.remove(temp_file)
                        logger.info(f"成功删除临时文件 (AIQ): {temp_file}")
                    except Exception as e_remove:
                        logger.warning(f"删除临时文件 {temp_file} 失败 (AIQ): {e_remove}")
    
    # AIQ 注册函数需要 yield FunctionInfo
    yield FunctionInfo.from_fn(
        _process_article,
        description="使用 AIQ 管理的 LLM、多个 Agno Agent、用于 Minimax TTS 的 MCPTools，将文章转换为语音，下载音频并进行合并。"
    )

