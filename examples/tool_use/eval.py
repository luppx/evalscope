import copy
import json
import os
import pdb
import requests
import shutil
import threading
import traceback
import uuid
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from openai import APIStatusError, BadRequestError, OpenAI, PermissionDeniedError, UnprocessableEntityError
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from transformers import AutoTokenizer

from evalscope import run_task
from evalscope.arguments import parse_args
from evalscope.api.messages.content import Content
from evalscope.api.messages import ChatMessage, ChatMessageTool
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput, ChatCompletionChoice, ModelUsage
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo, ToolParams
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from evalscope.models.utils.openai import (
    chat_choices_from_openai,
    collect_stream_response,
    model_output_from_openai,
    openai_chat_messages,
    openai_chat_tool_choice,
    openai_chat_tools,
    openai_completion_params,
    openai_handle_bad_request,
)
from evalscope.utils.logger import get_logger
from .code_sandbox import PythonSandbox
from .async_jupyter import JupyterTool, IPYTHONDIR

logger = get_logger()

# 1. 使用register_model_api注册模型
@register_model_api(name='server_tool_use')
class ToolUseModel(OpenAICompatibleAPI):
    """Tool use model API implementation."""
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Dict[str, Any],
    ) -> None:
        self.model_args = model_args
        logger.info(f"Generation config: {config.model_dump()}")
        logger.info(f"model_args: {self.model_args}")
        # 加载本地模型tokenizer，用于计算token数
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.get("local_path"))

        if model_args.get("local_path"):
            del model_args["local_path"]
        super().__init__(model_name, base_url, api_key, config, **model_args)

        # self.tools = [
        #     {
        #         "type": "function",
        #         "function": {
        #             "name": "python",
        #             "description": "Executes Python code in a stateless sandbox. The code must be a complete script with all necessary imports, and outputs in the code script must explicitly call the print() function.",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "code": {
        #                         "type": "string",
        #                         "description": 'The Python script to execute. It must include all required imports and use print() function to display any results.',
        #                     },
        #                 },
        #                 "required": ["code"],
        #             },
        #         },
        #     }
        # ]

        # 无状态沙盒
        # self.python_tool = PythonSandbox()
        # 有状态Jupyter
        self.python_tool = JupyterTool(concurrency=config.batch_size)
        self.default_tools = [self.python_tool.get_tool_info()]

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # pdb.set_trace()
        # 实现模型推理逻辑
        session_id = "generate_" + uuid.uuid4().hex
        tools = tools + self.default_tools if tools else self.default_tools
        if len(tools) > 0 and tool_choice == 'none':
            tool_choice = 'auto'
        # get completion params (slice off service from model name)
        completion_params = self.completion_params(
            config=config,
            tools=len(tools) > 0,
        )

        prompt_token_ids = self.get_token_ids(openai_chat_messages(input), openai_chat_tools(tools) if len(tools) > 0 else None)
        max_tokens = completion_params.get('max_tokens', None)
        used_tool = False
        completions = []
        try:
            while True:
                request = dict(
                    messages=openai_chat_messages(input),
                    tools=openai_chat_tools(tools) if len(tools) > 0 else NOT_GIVEN,
                    tool_choice=openai_chat_tool_choice(tool_choice) if len(tools) > 0 else NOT_GIVEN,
                    **completion_params,
                )

                if len(input) <= 1:
                    logger.info(f"[session_id: {session_id}] First round request: {request}")

                # update remain tokens
                if max_tokens is not None and len(input) > 1:
                    token_ids = self.get_token_ids(request["messages"], request["tools"])
                    remain_tokens = max_tokens - (len(token_ids) - len(prompt_token_ids))
                    logger.info(f"[session_id: {session_id}] max_tokens: {max_tokens}, prompt_tokens: {len(prompt_token_ids)}, "
                                f"cur_total_tokens: {len(token_ids)}, remain_tokens: {remain_tokens}")
                    if remain_tokens <= 0:
                        logger.warning(f"[session_id: {session_id}] Warning: remain tokens for generation is less or equal to 0, finish generation.")
                        # token数超长时直接返回，避免tool response太长导致输入给推理引擎的input过长
                        # 返回一个自定义的空completion，令其finish reason为length
                        completion = ChatCompletion(
                            id="exceed_length_" + uuid.uuid4().hex,
                            created=int(time.time()),
                            model=self.model_name,
                            object="chat.completion",
                            choices=[Choice(finish_reason="length", index=0, message=ChatCompletionMessage(role="assistant"))],
                        )
                        logger.info(f"[session_id: {session_id}] Construct a custom completion due to exceed max_tokens: {completion.model_dump()}")
                        completions.append(completion)
                        choices = self.chat_choices_from_completion(completion, tools)
                        logger.info(f"[session_id: {session_id}] choices due to exceed max_tokens: {choices}")
                        break
                    request['max_tokens'] = max(remain_tokens, 1)  # max_tokens must be >= 1
                
                # generate completion and save response for model call
                completion = self.client.chat.completions.create(**request)
                # handle streaming response
                if not isinstance(completion, ChatCompletion):
                    completion = collect_stream_response(completion)
                
                # 如果用vllm推理，使用的tool_call_parser有点问题(v0.10.1.1)，会固定在content前加两个\n，所以如果用vllm推理，要手动去除掉
                if len(completion.choices) > 0 and isinstance(completion.choices[0].message.content, str):
                    completion.choices[0].message.content = completion.choices[0].message.content.lstrip('\n\n')
                
                completions.append(completion)
                # logger.debug(f'[session_id: {session_id}] completion: {completion.model_dump()}')

                choices = self.chat_choices_from_completion(completion, tools)
                choice = choices[0]
                # 把所有输出都加到input中留存
                input.append(choice.message)

                if choice.stop_reason != 'tool_calls':
                    break
                if choice.message.tool_calls and not used_tool:
                    used_tool = True
                if tool_calls := choice.message.tool_calls:
                    tool_response_msgs = self.process_tool_calls(tool_calls, session_id)
                    input.extend(tool_response_msgs)

            return self.get_model_output(input, tools, completions, choices)

        except (BadRequestError, UnprocessableEntityError, PermissionDeniedError) as ex:
            return self.handle_bad_request(ex)
        
        finally:
            try:
                if used_tool:
                    self.python_tool.del_client(session_id)
            except Exception as e:
                logger.error(f"[session_id: {session_id}] Error occurred while deleting Jupyter client: {e}, "
                             f"traceback: {traceback.format_exc()}")

    def get_token_ids(self, messages: list, tools: list):
        return self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True)

    def process_tool_calls(self, tool_calls: List[Dict[str, Any]], session_id: str) -> List[ChatMessageTool]:
        tool_response_msgs = []
        for tool_call in tool_calls:
            if tool_call.parse_error:
                msg = ChatMessageTool(
                    tool_call_id=tool_call.id,
                    function=tool_call.function.name,
                    error=tool_call.parse_error,
                )
                tool_response_msgs.append(msg)
                continue
            
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments
            try:
                fn = self._get_function_by_name(fn_name)
            except Exception as e:
                logger.error(f"[session_id: {session_id}] Error getting function {fn_name}: {e}")
                msg = ChatMessageTool(
                    tool_call_id=tool_call.id,
                    function=fn_name,
                    error=e,
                )
                tool_response_msgs.append(msg)
                continue

            # 如果fn_args参数不正确，把logger打印出来，方便调试
            if set(fn_args.keys()) != {'code'}:
                logger.error(f"[session_id: {session_id}] Invalid tool call argument. Tool call: {tool_call}, fn_args: {fn_args}")
            
            # fn_res = json.dumps(fn(**fn_args))
            fn_res = fn(session_id=session_id, **fn_args)

            msg = ChatMessageTool(
                tool_call_id=tool_call.id,
                function=fn_name,
                content=fn_res,
            )
            tool_response_msgs.append(msg)
        
        return tool_response_msgs

    def _get_function_by_name(self, name):
        if name == "python":
            return self._python
        else:
            raise ValueError(f"Error: Tool '{name}' not found")
    
    def _python(self, session_id: str, code: str) -> str:
        return self.python_tool.execute_code(session_id, code)

    def get_model_output(
        self,
        messages: List[ChatMessage],
        tools: List[ToolInfo],
        completions: List[ChatCompletion],
        choices: List[ChatCompletionChoice],
    ) -> ModelOutput:
        msgs = copy.deepcopy(messages)
        if choices:
            msgs.append(choices[0].message)
        msgs = openai_chat_messages(msgs)
        tools_list = openai_chat_tools(tools) if len(tools) > 0 else None
        prompt_token_ids = self.get_token_ids([msgs[0]], tools_list)
        input_tokens = len(prompt_token_ids) if prompt_token_ids is not None else None

        total_token_ids = self.get_token_ids(msgs, tools_list)
        total_tokens = len(total_token_ids) if total_token_ids is not None else None

        output_tokens = total_tokens - input_tokens if total_tokens is not None and input_tokens is not None else None

        return ModelOutput(
            model=completions[-1].model,
            choices=choices,
            usage=(
                ModelUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                ) if input_tokens or output_tokens or total_tokens else None
            ),
        )

        # input_tokens = completions[0].usage.prompt_tokens if completions[0].usage is not None else None
        
        # output_tokens = 0
        # for completion in completions:
        #     if completion.usage is None:
        #         output_tokens = None
        #         break
        #     output_tokens += completion.usage.completion_tokens
        
        # input_tokens_cache_read = (
        #     completions[0].usage.prompt_tokens_details.cached_tokens if completions[0].usage is not None 
        #     and completions[0].usage.prompt_tokens_details is not None else None  # openai only have cache read stats/pricing.
        # )

        # reasoning_tokens = 0
        # for completion in completions:
        #     if completion.usage is None or completion.usage.completion_tokens_details is None:
        #         reasoning_tokens = None
        #         break
        #     reasoning_tokens += completion.usage.completion_tokens_details.reasoning_tokens

        # total_tokens = completions[-1].usage.total_tokens if completions[-1].usage is not None else None

        # return ModelOutput(
        #     model=completions[-1].model,
        #     choices=choices,
        #     usage=(
        #         ModelUsage(
        #             input_tokens=input_tokens,
        #             output_tokens=output_tokens,
        #             input_tokens_cache_read=input_tokens_cache_read,
        #             reasoning_tokens=reasoning_tokens,
        #             total_tokens=total_tokens,
        #         ) if input_tokens or output_tokens or input_tokens_cache_read or reasoning_tokens or total_tokens else None
        #     ),
        # )


if __name__ == '__main__':
    args = parse_args()
    start_time = datetime.now()
    logger.info(f"Evaluation started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    result = run_task(args)
    end_time = datetime.now()
    cost_time = (end_time - start_time).total_seconds()
    logger.info(f"Evaluation finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, took {cost_time / 60} minutes, {cost_time / 3600} hours")
    
    if os.path.isdir(IPYTHONDIR):
        logger.info(f"Removing IPYTHONDIR: {IPYTHONDIR}")
        shutil.rmtree(IPYTHONDIR, ignore_errors=True)