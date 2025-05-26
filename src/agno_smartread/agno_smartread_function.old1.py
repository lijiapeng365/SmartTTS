# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pathlib
import json
from textwrap import dedent
from typing import List, Dict, Any
import uuid
# import re # 如果TTS_Agent最终输出不是纯JSON，可能需要

from aiq.builder.builder import Builder # AIQ Builder 仍然用于获取LLM
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef # FunctionRef不再直接用于MCP工具
from aiq.data_models.function import FunctionBaseConfig

# >>> 新增：导入Agno的MCPTools <<<
from agno.tools.mcp import MCPTools

from pydub import AudioSegment

logger = logging.getLogger(__name__)


# --- >>> 修改配置类 <<< ---
class ArticleToSpeechFunctionConfig(FunctionBaseConfig, name="article_to_speech_workflow"):
    llm_name: LLMRef
    mcp_server_url: str # 新增：用于Agno MCPTools的服务器URL
    temp_audio_directory: str = "./temp_audio_files"


@register_function(config_type=ArticleToSpeechFunctionConfig, framework_wrappers=[LLMFrameworkEnum.AGNO])
async def article_to_speech_workflow(config: ArticleToSpeechFunctionConfig, builder: Builder):
    from agno.agent import Agent

    os.makedirs(config.temp_audio_directory, exist_ok=True)
    temp_audio_dir_literal = config.temp_audio_directory

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.AGNO)

    text_split_agent = Agent(
        name="text_split_agent",
        role="Based on the input article content, intelligently split the article into different sections by character, or identify it as a single-speaker document.",
        model=llm, 
        description=dedent("""\
        Analyzes input text. If multiple characters and dialogue are present, identifies them and outputs a JSON list with 'character_id' and 'text_segment', merging segments and assigning narration to 'Narrator', strictly attributing only quoted text to characters.
        If the text appears to be from a single speaker or has no dialogue (e.g., news report, announcement), it categorizes the entire text under a single 'character_id' (typically 'Narrator' or a specific speaker if identifiable).
        """),
        instructions=[ 
            "You will be given an article text.",
            "Your primary task is to analyze the text and determine if it contains multiple speaking characters and dialogue, or if it is a single-speaker document (e.g., a news report, a speech, a technical article, a first-person narrative without dialogue with others).",
            "**Scenario 1: Multiple Characters and Dialogue Present**",
            "  If the article contains dialogue from multiple characters:",
            "    Identify all characters mentioned by name who are speaking.",
            "    **Crucially, only the exact text enclosed in quotation marks (e.g., \"dialogue\" or 'dialogue') should be attributed to a speaking character as their 'text_segment'.**",
            "    All other text, including descriptions of who is speaking (e.g., 'Mary said softly,'), actions, scene descriptions, and general narration, MUST be attributed to the 'Narrator' character_id.",
            "    If multiple consecutive text segments belong to the 'Narrator', merge them into a single 'text_segment' for the 'Narrator'.",
            "    The final output MUST be a valid JSON list of objects.",
            "    Each object in the list must contain two keys: 'character_id' (string) and 'text_segment' (string).",
            "    Ensure that the 'character_id' for speaking characters is derived consistently from their names. (e.g., 'Mr. Henderson' to 'Mr_Henderson').",
            "    Here is an example of the input article format and the desired output format for Scenario 1 (strictly quoted text for characters):",
            """
            Example Input Article Text for Scenario 1:
            The sun dipped below the horizon, painting the sky in hues of orange and purple.
            "What a beautiful sunset," Mary said softly, leaning against the railing.
            John nodded beside her. "It is. Makes you forget all the troubles of the day." He sighed. "I wish these moments could last forever."
            Mary turned to him. "They do, in a way. In our memories." She smiled.
            The old lighthouse keeper, Mr. Henderson, walked by, tipping his hat. "Evenin', folks. Don't stay out too late, the tide's coming in."
            "Thanks, Mr. Henderson!" John called back.
            "We should head back soon," Mary agreed.

            Desired JSON Output for the Scenario 1 example above (Strictly quoted text for characters):
            [
              {
                "character_id": "Narrator",
                "text_segment": "The sun dipped below the horizon, painting the sky in hues of orange and purple."
              },
              {
                "character_id": "Narrator",
                "text_segment": "Mary said softly, leaning against the railing."
              },
              {
                "character_id": "Mary",
                "text_segment": "\\"What a beautiful sunset,\\""
              },
              {
                "character_id": "Narrator",
                "text_segment": "John nodded beside her."
              },
              {
                "character_id": "John",
                "text_segment": "\\"It is. Makes you forget all the troubles of the day.\\""
              },
              {
                "character_id": "Narrator",
                "text_segment": "He sighed."
              },
              {
                "character_id": "John",
                "text_segment": "\\"I wish these moments could last forever.\\""
              },
              {
                "character_id": "Narrator",
                "text_segment": "Mary turned to him."
              },
              {
                "character_id": "Mary",
                "text_segment": "\\"They do, in a way. In our memories.\\""
              },
              {
                "character_id": "Narrator",
                "text_segment": "She smiled. The old lighthouse keeper, Mr. Henderson, walked by, tipping his hat."
              },
              {
                "character_id": "Mr_Henderson",
                "text_segment": "\\"Evenin', folks. Don't stay out too late, the tide's coming in.\\""
              },
              {
                "character_id": "Narrator",
                "text_segment": "John called back."
              },
              {
                "character_id": "John",
                "text_segment": "\\"Thanks, Mr. Henderson!\\""
              },
              {
                "character_id": "Narrator",
                "text_segment": "Mary agreed."
              },
              {
                "character_id": "Mary",
                "text_segment": "\\"We should head back soon,\\""
              }
            ]
            """,
            "**Scenario 2: Single Speaker or No Dialogue Article**",
            "  If you determine the article does not contain dialogue between multiple characters, or is a monologue, news report, announcement, or similar single-perspective text:",
            "    You should output a JSON list containing a SINGLE object.",
            "    This object should have a 'character_id'. This 'character_id' should be 'Narrator' if no specific speaker is identifiable from the text itself (e.g., a general news article). If a specific speaker is clearly identifiable as the sole voice (e.g., 'Announcer:', 'From the CEO's Desk:'), you may use that identifier (e.g., 'Announcer', 'CEO'). For most generic cases, 'Narrator' is appropriate.",
            "    The 'text_segment' for this single object should be the ENTIRE, UNMODIFIED original article text.",
            "    Do NOT attempt to split the text in this scenario.",
            "    Here is an example for Scenario 2 (Single Speaker/No Dialogue):",
            """
            Example Input Article Text for Scenario 2 (News Report):
            The city council today announced new measures to improve public transportation. The initiative, set to roll out next spring, will include the addition of 30 new electric buses and an overhaul of the existing metro schedule. Councilmember Ann Li stated that these changes aim to reduce traffic congestion and promote greener travel options for all citizens. The project is budgeted at $15 million.

            Desired JSON Output for the Scenario 2 example above:
            [
              {
                "character_id": "Narrator",
                "text_segment": "The city council today announced new measures to improve public transportation. The initiative, set to roll out next spring, will include the addition of 30 new electric buses and an overhaul of the existing metro schedule. Councilmember Ann Li stated that these changes aim to reduce traffic congestion and promote greener travel options for all citizens. The project is budgeted at $15 million."
              }
            ]
            """,
            "**Output Format Reminder:**",
            "Always output a valid JSON list. This list will contain multiple objects for Scenario 1, and a single object for Scenario 2.",
            "Now, process the provided article text based on these instructions, first determining if it's a multi-dialogue scenario or a single-speaker/no-dialogue scenario.",
            "IMPORTANT FINAL INSTRUCTION: Your entire response must be ONLY the valid JSON list itself, starting with '[' and ending with ']'. Do NOT include any other text, explanations, conversation, or markdown code block formatting (like ```json) surrounding the JSON data. Provide only the raw JSON content."
        ]
    )

    async def _arun(article_file_path: str) -> str:
        article_text: str
        logger.info(f"Raw input for workflow: {article_file_path!r}")
        try: # Outer try for file reading
            logger.info(f"Attempting to read article from file: {article_file_path}")
            input_file = pathlib.Path(article_file_path)
            if not input_file.is_file():
                logger.error(f"Input file not found: {article_file_path}")
                return json.dumps({"final_audio_path": None, "status": "failure", "details": f"Input file not found: {article_file_path}"})

            article_text = input_file.read_text(encoding="utf-8")
            logger.info(f"Successfully read {len(article_text)} characters from file: {article_file_path}")

            if not article_text.strip():
                logger.warning(f"Input article file '{article_file_path}' is empty or contains only whitespace.")
                return json.dumps({"final_audio_path": None, "status": "success", "details": f"Input article file '{article_file_path}' is empty."})

        except Exception as e_file_read: # Catch file reading specific errors
            logger.error(f"Error reading article file {article_file_path}: {e_file_read}")
            return json.dumps({"final_audio_path": None, "status": "failure", "details": f"Error reading input file: {e_file_read}"})
        
        # --- >>> Main logic block wrapped in a try for e_outer <<< ---
        try:
            async with MCPTools(url=config.mcp_server_url, transport="streamable-http") as mcp_tools_obj:
                logger.info(f"MCPTools initialized for server: {config.mcp_server_url}")
                
                tts_agent_description_str = dedent(f"""\
                This agent takes a list of text segments, each associated with a character_id, as input.
                It first calls the `list_voices` tool to discover available voices from the MCP server.
                Then, for each text segment, it intelligently assigns a voice and calls the `text_to_audio` tool
                to generate an audio file. Each generated audio file is saved to a unique path within the
                '{temp_audio_dir_literal}' directory.
                The agent's final output is a single JSON object containing two lists:
                'generated_audio_files' (paths of successfully created .mp3 files) and
                'failed_segments_info' (details of segments that could not be processed).
                When calling a tool, the agent will output a string in the format: tool_name(param1="value1", param2="value2").
                """)

                tts_agent_instructions_list = [
                    f"You are an AI assistant that generates multiple audio files from text segments by calling tools and then reports the outcome.",
                    f"You will be given a JSON list of text segments as input. Each item has 'character_id' and 'text_segment'. This is your primary task context.",
                    f"You have access to tools provided by an MCP server. The key tools are `list_voices` and `text_to_audio`.",
                    f"To call a tool, your entire response for that turn MUST BE ONLY a string representing the function call.",
                    f"  - For `list_voices` (which takes no parameters), output exactly: `list_voices()`",
                    f"  - For `text_to_audio` (which takes `text`, `voice_id`, `outputFile` parameters), output in this format: `text_to_audio(text=\"<actual_text>\", voice_id=\"<actual_voice_id>\", outputFile=\"<actual_output_file_path>\")` (ensure string values within the call are properly quoted, especially the text parameter).",
                    f"Do NOT output JSON when you are calling a tool. Only output the function call string.",
                    f"**Follow this multi-step process:**",
                    f"**STEP 1: Get Available Voices**",
                    f"1. Your first action should be to call the `list_voices` tool. Output the function call string for it.",
                    f"2. The system will execute this tool and provide you with its output (assumed to be a list of voice ID strings, e.g., `['voice_alpha', 'voice_beta']`). If the tool call fails or returns unusable data, you may use a default list like `['voice_default_1', 'voice_default_2']` for subsequent steps, but try to use the actual tool output if provided.",
                    f"**STEP 2: Process Each Text Segment to Generate Audio (After receiving voice list)**",
                    f"Once you have the voice list (either from the tool or your default), you need to process the initial list of text segments.",
                    f"Initialize an empty list in your internal state called `generated_audio_files_list`.",
                    f"Initialize an empty list in your internal state called `failed_segments_info_list`.",
                    f"Iterate through the *initial* input list of segments (provided at the very beginning of our interaction). For each segment object (with 'character_id', 'text_segment') at index `i` (0-indexed):",
                    f"   a. Let `current_char_id = segment_object['character_id']`.",
                    f"   b. Let `current_text = segment_object['text_segment']`.",
                    f"   c. **Input Validation:** If `current_text` is empty or whitespace, add an entry to `failed_segments_info_list` (e.g., `{{\"index\": i, \"character_id\": current_char_id, \"text_segment\": current_text, \"reason\": \"Text is empty\"}}`) and skip to the next segment. Do NOT call any tool for this segment.",
                    f"   d. **Assign Voice:** Select a `voice_id` for `current_char_id` from the voice list you have. Assign consistently for 'Narrator'. Cycle if needed.",
                    f"   e. **Determine Output File Path:** Construct the full `outputFile` path. The path MUST be in `{temp_audio_dir_literal}`. Use a pattern like `{temp_audio_dir_literal}/segment_INDEX_CHARID.mp3` (e.g., `{temp_audio_dir_literal}/segment_000_Narrator.mp3`). Ensure CHARID is filename-safe (underscores for spaces/special chars).",
                    f"   f. **Call Text-to-Audio Tool:** Output the function call string for `text_to_audio` with the correctly filled parameters `text`, `voice_id`, and `outputFile`.",
                    f"   g. **Await System Feedback for Each TTS Call:** After you request a `text_to_audio` call, the system will execute it and provide you its result (e.g., success, or an error message).",
                    f"      - If the tool call was successful (file created), add the `outputFile` path (that you sent to the tool) to your `generated_audio_files_list`.",
                    f"      - If the tool call failed, add an entry to `failed_segments_info_list` including the reason if provided by the system.",
                    f"**STEP 3: Final Output Generation**",
                    f"After you believe you have iterated through all segments and attempted audio generation for each (and received feedback for each TTS tool call):",
                    f"Your *sole and final response* MUST be a single, valid JSON object. This JSON object must contain two keys:",
                    f"1. `generated_audio_files`: Your populated list of successful .mp3 file paths.",
                    f"2. `failed_segments_info`: Your populated list of failure details.",
                    f"**Example final JSON output:**",
                    "`{{\"generated_audio_files\": [\"" + temp_audio_dir_literal + "/segment_000_Narrator.mp3\"], \"failed_segments_info\": [{\"index\": 1, ...}]}}`",
                    "Current datetime is: {{datetime}}",
                    "Begin processing now. Remember: when calling a tool, output ONLY the function call string. When finished with ALL segments and ALL tool calls, output ONLY your final summary JSON as described in STEP 3. Do not mix tool calls and final answers in one response."
                ]

                TTS_Agent = Agent(
                    name="TTS_Audio_Generator_MCP",
                    role="Processes segmented text, calls MCP tools (using string format) to list voices and generate audio, and reports results.",
                    model=llm,
                    description=tts_agent_description_str,
                    instructions=tts_agent_instructions_list,
                    tools=[mcp_tools_obj], 
                    add_datetime_to_instructions=True,
                )

                # Inner try for agent execution and subsequent audio processing
                try:
                    logger.info("Starting text segmentation...") # This log was outside the inner try, moved it for consistency
                    text_split_response = await text_split_agent.arun(article_text, stream=False)
                    text_split_content_str = (text_split_response.content
                                            if hasattr(text_split_response, 'content') else str(text_split_response))
                    if not text_split_content_str:
                        logger.error("Text split agent returned no content.")
                        return json.dumps({"final_audio_path": None, "status": "failure", "details": "Text splitting failed to produce output."})
                    logger.debug(f"Text split raw response: \n {text_split_content_str}")
                    try:
                        segmented_texts_list = json.loads(text_split_content_str)
                    except json.JSONDecodeError as e_json_split:
                        logger.error(f"Failed to decode JSON from text_split_agent: {e_json_split}")
                        logger.error(f"Invalid JSON string was: {text_split_content_str}")
                        return json.dumps({"final_audio_path": None, "status": "failure", "details": f"Text splitting produced invalid JSON: {e_json_split}"})
                    if not isinstance(segmented_texts_list, list) or not all(isinstance(item, dict) for item in segmented_texts_list) :
                        logger.error(f"Text split agent did not return a list of dicts as expected. Got: {type(segmented_texts_list)}")
                        return json.dumps({"final_audio_path": None, "status": "failure", "details": "Text splitting output was not in the expected format (list of dicts)."})
                    if not segmented_texts_list:
                        logger.info("Text split agent returned an empty list of segments. Nothing to process for TTS.")
                        return json.dumps({"final_audio_path": None, "status": "success", "details": "No text segments to convert to speech."})

                    logger.info(f"Handing off {len(segmented_texts_list)} segments to TTS_Agent (using Agno MCPTools) for internal tool execution...")
                    tts_input_payload_str = json.dumps(segmented_texts_list)
                    
                    tts_agent_response_obj = await TTS_Agent.arun(tts_input_payload_str, stream=False)

                    tts_agent_content_str = (tts_agent_response_obj.content
                                             if hasattr(tts_agent_response_obj, 'content') else str(tts_agent_response_obj))
                    
                    print(f"TTS_Agent RAW OUTPUT (expecting final JSON summary or error from Agno MCPTools): \n{tts_agent_content_str}")

                    if not tts_agent_content_str:
                        logger.error("TTS agent returned no content after processing segments and tools.")
                        return json.dumps({"final_audio_path": None, "status": "failure", "details": "TTS agent failed to produce output after tool execution."})

                    try:
                        tts_results = json.loads(tts_agent_content_str)
                        executed_audio_files = tts_results.get("generated_audio_files", [])
                        failed_segments_info = tts_results.get("failed_segments_info", [])
                    except json.JSONDecodeError as e_json_tts:
                        logger.error(f"Failed to decode FINAL JSON summary from TTS_Agent: {e_json_tts}")
                        logger.error(f"Invalid FINAL JSON string from TTS_Agent was: {tts_agent_content_str}")
                        return json.dumps({"final_audio_path": None, "status": "failure", "details": f"TTS_Agent's final output was not the expected JSON summary. Output: {tts_agent_content_str}"})

                    if failed_segments_info:
                        for failure in failed_segments_info:
                            logger.warning(f"TTS Agent reported failure for segment index {failure.get('index', 'N/A')}, char: {failure.get('character_id', 'N/A')}. Reason: {failure.get('reason', 'Unknown')}")
                    
                    if not executed_audio_files:
                        logger.warning("TTS Agent reported no audio files were successfully generated in its final summary.")
                        details_msg = "TTS Agent's final summary reported no audio files generated."
                        if failed_segments_info:
                            details_msg += " Some segments reportedly failed during TTS processing."
                        return json.dumps({"final_audio_path": None, "status": "success" if not segmented_texts_list else "failure",
                                            "details": details_msg})

                    logger.info(f"Concatenating {len(executed_audio_files)} audio files...")
                    combined_audio = AudioSegment.empty()
                    successfully_loaded_segments = 0
                    for audio_file_path_loop_var in executed_audio_files: 
                        if not os.path.exists(audio_file_path_loop_var):
                            logger.error(f"Audio file specified by TTS agent does not exist: {audio_file_path_loop_var}")
                            continue
                        try:
                            file_format = pathlib.Path(audio_file_path_loop_var).suffix[1:].lower() 
                            if not file_format: 
                                logger.warning(f"No file extension found for {audio_file_path_loop_var}. Assuming mp3.")
                                file_format = "mp3"
                            elif file_format not in ["mp3", "wav", "ogg", "flv", "aac"]: 
                                logger.warning(f"Unsupported audio format '{file_format}' for Pydub in file {audio_file_path_loop_var}. Defaulting to mp3.")
                                file_format = "mp3" 
                            
                            segment_audio = AudioSegment.from_file(audio_file_path_loop_var, format=file_format)
                            combined_audio += segment_audio
                            successfully_loaded_segments += 1
                        except Exception as e_concat:
                            logger.error(f"Error loading or concatenating audio file {audio_file_path_loop_var}: {e_concat}")
                            continue
                    
                    if successfully_loaded_segments == 0 and len(executed_audio_files) > 0 :
                        logger.error("No audio segments could be loaded for concatenation, though TTS agent reported generated files.")
                        return json.dumps({"final_audio_path": None, "status": "failure", "details": "Failed to load any generated audio segments for concatenation."})

                    if len(combined_audio) > 0:
                        run_specific_id = str(uuid.uuid4())[:8]
                        final_audio_filename = f"final_article_speech_{run_specific_id}.mp3"
                        final_audio_path_val = os.path.join(temp_audio_dir_literal, final_audio_filename)
                        
                        try:
                            combined_audio.export(final_audio_path_val, format="mp3")
                            logger.info(f"Final combined audio saved to: {final_audio_path_val}")
                        except Exception as e_export:
                            logger.error(f"Failed to export combined audio to {final_audio_path_val}: {e_export}")
                            return json.dumps({"final_audio_path": None, "status": "failure", "details": f"Failed to save final audio: {e_export}"})
                                                
                        return json.dumps({"final_audio_path": final_audio_path_val, "status": "success", 
                                            "details": f"{successfully_loaded_segments}/{len(executed_audio_files)} segments successfully processed and combined. Check logs for any TTS or tool execution failures."})
                    else:
                        status_detail_val = "No audio segments were successfully generated or loaded to concatenate."
                        if not executed_audio_files: 
                            status_detail_val = "No audio segments were reported as generated by TTS."
                        
                        return json.dumps({"final_audio_path": None, "status": "success" if not executed_audio_files and not failed_segments_info else "failure", 
                                            "details": status_detail_val})

                except Exception as e_inner_arun: 
                    logger.exception(f"Error during TTS Agent execution or subsequent processing: {str(e_inner_arun)}")
                    return json.dumps({"final_audio_path": None, "status": "failure", "details": f"Error during TTS Agent execution: {str(e_inner_arun)}"})
            # If an error occurred in `async with MCPTools(...)` itself, it will be caught by e_outer
            # Or if the inner try-except returns, this point is not reached.

        except Exception as e_outer: # This now correctly catches errors from the main logic block
            logger.exception(f"Critical error in article_to_speech_workflow _arun function's main processing: {str(e_outer)}")
            return json.dumps({"final_audio_path": None, "status": "failure", "details": f"An unexpected critical error occurred in main logic: {str(e_outer)}"})

    yield FunctionInfo.from_fn(_arun, description="Converts an article to multi-role speech audio by splitting text, having an agent generate TTS by calling Agno MCP tools, and then concatenating audio.")
