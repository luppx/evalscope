import asyncio
import os
import threading
import traceback
from jupyter_client import AsyncKernelManager, AsyncKernelClient
from jupyter_client.manager import start_new_async_kernel
from queue import Empty
from typing import Any, Dict, List, Optional
from evalscope.api.tool import ToolInfo, ToolParams
from evalscope.utils.logger import get_logger

logger = get_logger()

JUPYTER_CLIENTS: Dict[str, "AsyncJupyterClient"] = {}
IPYTHONDIR = "/tmp/ipython"
os.makedirs(IPYTHONDIR, exist_ok=True)

class AsyncJupyterClient:
    def __init__(self, session_id: str, km: AsyncKernelManager, kc: AsyncKernelClient, timeout: int = 120):
        self.session_id = session_id
        self.timeout = timeout
        self.km = km
        self.kc = kc

    async def end_session(self):
        if self.kc:
            self.kc.stop_channels()
        if self.km:
            await self.km.shutdown_kernel()

    # async def execute(self, code: str) -> str:
    #     msg_id = self.kc.execute(code)
    #     msg = None  # 防止未定义
    #     try:
    #         while True:
    #             try:
    #                 msg = await self.kc.get_iopub_msg(timeout=self.timeout)
    #             except Empty:
    #                 logger.error(f"[session_id: {self.session_id}] Timeout waiting for Jupyter message. code: '{code}'\ntraceback: {traceback.format_exc()}")
    #                 return f"Execution timeout after {self.timeout} seconds."
    #             if msg['parent_header'].get('msg_id') == msg_id:
    #                 if msg['msg_type'] == 'stream':
    #                     return msg['content']['text']
    #                 elif msg['msg_type'] == 'error':
    #                     return '\n'.join(msg['content']['traceback'])
    #                 elif msg['msg_type'] == 'execute_result':
    #                     return str(msg['content']['data'].get('text/plain', ''))
    #                 elif msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
    #                     return ""
    #     except Exception as e:
    #         err = f"[session_id: {self.session_id}] Error occurred during Jupyter notebook code execution. code: '{code}'\nerror: {e}\ntraceback: {traceback.format_exc()}"
    #         if msg:
    #             err += f"\nmsg: {msg}"
    #         logger.error(err)
    #         raise e

    # Reference SandboxFusion's implementation: https://github.com/bytedance/SandboxFusion/blob/main/runtime/jupyter/main.py#L67
    async def execute(self, code: str) -> str:
        try:
            cell_result = {
            'output': [],
            'error': [],
            }
            def hook(msg):
                # logger.info(f"[session_id: {self.session_id}] msg_type: {msg_type}, content: {content}")
                msg_type = msg["header"]["msg_type"]
                content = msg["content"]
                if msg_type == "stream":
                    if content["name"] == "stdout":
                        cell_result["output"].append(content["text"])
                    elif content["name"] == "stderr":
                        cell_result["error"].append(content["text"])
                    else:
                        cell_result[content["name"]] += content["text"]
                elif msg_type in ("display_data", "execute_result"):
                    cell_result["output"].append(content["data"])
                elif msg_type == "error":
                    cell_result['error'].append(content)
            try:
                result = await self.kc.execute_interactive(code, output_hook=hook, timeout=self.timeout)
            except TimeoutError as te:
                logger.error(f"[session_id: {self.session_id}] Timeout waiting for Jupyter message. code: '{code}'\nerror: {te}\ntraceback: {traceback.format_exc()}")
                logger.info(f"[session_id: {self.session_id}] Interrupting kernel due to timeout.")
                await self.km.interrupt_kernel()
                logger.info(f"[session_id: {self.session_id}] Kernel interrupted.")
                return f"Execution timeout after {self.timeout} seconds."
            status = result['content']['status']
            cell_result['status'] = status
            logger.debug(f"[session_id: {self.session_id}] Jupyter execution result: {cell_result}")

            # status: https://jupyter-client.readthedocs.io/en/stable/messaging.html#request-reply
            if status == 'ok':
                res = []
                for display in cell_result['output']:
                    if isinstance(display, str): # stdout
                        res.append(display)
                        continue
                    for mime, data in display.items():
                        data = data + "\n"
                        res.append(data)
                return "".join(res) if res else ""
            elif status == 'error':    
                errors = []
                for error in cell_result['error']:
                    if isinstance(error, str):
                        errors.append(error)
                    elif isinstance(error, dict):
                        tb_lines = error.get('traceback', [])
                        for line in tb_lines:
                            line = line + "\n"
                            errors.append(line)
                return "".join(errors) if errors else "An error occurred during code execution, but no error details were captured."
            elif status == 'aborted':
                return "Execution was aborted."
            else:
                raise RuntimeError(f"Unknown execution status: {status}")
        except Exception as e:
            err = f"[session_id: {self.session_id}] Error occurred during Jupyter notebook code execution. code: '{code}'\nerror: {e}\ntraceback: {traceback.format_exc()}"
            logger.error(err)
            raise e

# async def get_or_create_client_async(session_id: str, timeout: int = 120) -> AsyncJupyterClient:
#     async with JUPYTER_CLIENTS_LOCK:
#         client = JUPYTER_CLIENTS.get(session_id)
#         if client:
#             logger.info(f"Jupyter session {session_id} retrieved from global clients.")
#             return client

#         logger.info(f"Creating new Jupyter session {session_id}.")
#         try:
#             # 给每个kernel分配一个单独的目录维护history db等数据，避免多个kernel竞争写锁冲突，报错[IPKernelApp] ERROR | Failed to create history session
#             km, kc = await start_new_async_kernel(
#                 kernel_name="python3",
#                 env={
#                     "IPYTHONDIR": os.path.join(IPYTHONDIR, f"ipython_{session_id}"),
#                     # 下面这两个环境变量没用，每个kernel还是会创建自己的目录
#                     "IPYTHON_HISTORY_FILE": ":memory:",
#                     "IPYTHON_NO_HISTORY": "1",
#                     **os.environ,
#                 }
#             )
#         except Exception as e:
#             err = f"Error occurred while starting Jupyter session {session_id}: {e}"
#             logger.error(err)
#             raise RuntimeError(err) from e

#         client = AsyncJupyterClient(session_id=session_id, km=km, kc=kc, timeout=timeout)
#         JUPYTER_CLIENTS[session_id] = client
#         return client

# async def del_client_async(session_id: str):
#     async with JUPYTER_CLIENTS_LOCK:
#         client = JUPYTER_CLIENTS.pop(session_id, None)
#     # 锁外关闭，避免长时间占用锁
#     if client:
#         await client.end_session()
#         logger.info(f"Jupyter session {session_id} deleted from global clients.")
#     else:
#         logger.warning(f"Jupyter session {session_id} not found in global clients.")


class JupyterTool:
    def __init__(self, timeout: int = 120, concurrency: int = 32):
        self.concurrency = concurrency
        self.timeout = timeout
        self.set_tool_info()

        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self.semaphore = None
        self.clients_lock = None  # Lock 将在 loop 内创建
        # 在后台事件循环中创建 semaphore 和 lock
        self._initialize_async_components()
        assert self.semaphore is not None, "Semaphore initialization failed"
        assert self.clients_lock is not None, "Clients lock initialization failed"

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _initialize_async_components(self):
        """在后台 loop 中创建 Semaphore 和 Lock"""
        async def init():
            self.semaphore = asyncio.Semaphore(self.concurrency)
            self.clients_lock = asyncio.Lock()
        fut = asyncio.run_coroutine_threadsafe(init(), self.loop)
        fut.result()  # 等待初始化完成

    async def _get_or_create_client(self, session_id: str) -> AsyncJupyterClient:
        """使用 self.clients_lock 控制并发访问"""
        async with self.clients_lock:
            client = JUPYTER_CLIENTS.get(session_id)
            if client:
                logger.info(f"Jupyter session {session_id} retrieved from global clients.")
                return client

            logger.info(f"Creating new Jupyter session {session_id}.")
            try:
                # 给每个kernel分配一个单独的目录维护history db等数据，避免多个kernel竞争写锁冲突，报错[IPKernelApp] ERROR | Failed to create history session
                km, kc = await start_new_async_kernel(
                    kernel_name="python3",
                    env={
                        "IPYTHONDIR": os.path.join(IPYTHONDIR, f"ipython_{session_id}"),
                        # 下面这两个环境变量没用，每个kernel还是会创建自己的目录
                        "IPYTHON_HISTORY_FILE": ":memory:",
                        "IPYTHON_NO_HISTORY": "1",
                        **os.environ,
                    }
                )
            except Exception as e:
                err = f"Error occurred while starting Jupyter session {session_id}: {e}"
                logger.error(err)
                raise RuntimeError(err) from e

            client = AsyncJupyterClient(session_id=session_id, km=km, kc=kc, timeout=self.timeout)
            JUPYTER_CLIENTS[session_id] = client
            return client

    async def _del_client_async(self, session_id: str):
        async with self.clients_lock:
            client = JUPYTER_CLIENTS.pop(session_id, None)
        # 锁外关闭，避免长时间占用锁
        if client:
            await client.end_session()
            logger.info(f"Jupyter session {session_id} deleted from global clients.")
        else:
            logger.warning(f"Jupyter session {session_id} not found in global clients.")

    def del_client(self, session_id: str):
        """同步接口：将任务提交到后台 event loop"""
        fut = asyncio.run_coroutine_threadsafe(
            self._del_client_async(session_id),
            self.loop
        )
        return fut.result()  # 阻塞等待结果

    def set_tool_info(self):
        self.tool_info = ToolInfo(
            name="python",
            description="""
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. Internet access for this session is UNAVAILABLE.
        """.strip(),
            parameters=ToolParams(
                properties={
                    "code": {
                        "type": "string",
                        "description": 'The Python code to execute.',
                    },
                },
                required=["code"],
            ),
        )
    
    def get_tool_info(self) -> ToolInfo:
        return self.tool_info
    
    async def call_jupyter(self, session_id: str, code: str) -> str:
        logger.info(f"[session_id: {session_id}] code: {code}")
        async with self.semaphore:
            try:
                client = await self._get_or_create_client(session_id)
                output = await client.execute(code)
                logger.info(f"[session_id: {session_id}] output: {output}")
                return output
            except Exception as e:
                err = f"Exception occurred during Jupyter notebook code execution: {e}"
                logger.error(f"[session_id: {session_id}] {err}\ntraceback: {traceback.format_exc()}")
                return err
    
    def execute_code(self, session_id: str, code: str) -> str:
        """同步接口：将任务提交到后台 event loop"""
        fut = asyncio.run_coroutine_threadsafe(
            self.call_jupyter(session_id, code),
            self.loop
        )
        return fut.result()  # 阻塞等待结果
    
    def close(self):
        """关闭后台事件循环和线程"""
        async def shutdown():
            # 关闭所有 Jupyter 客户端
            async with self.clients_lock:
                session_ids = list(JUPYTER_CLIENTS.keys())
            for sid in session_ids:
                await self._del_client_async(sid)
        
        try:
            # 事件循环可能已不在运行（进程退出/析构阶段）
            if self.loop and self.loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(shutdown(), self.loop)
                fut.result()  # 等待关闭完成
        except Exception:
            logger.error(f"Error during JupyterTool shutdown: {traceback.format_exc()}")
        finally:
            try:
                if self.loop and self.loop.is_running():
                    self.loop.call_soon_threadsafe(self.loop.stop)
                if self._thread.is_alive():
                    self._thread.join(timeout=60)
            except Exception:
                logger.error(f"Error stopping JupyterTool loop/thread: {traceback.format_exc()}")
    
    # 依赖垃圾回收机制，析构时自动调用
    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.error(f"Exception in JupyterTool __del__: {e}. traceback: {traceback.format_exc()}")
    
    # 提供上下文管理器 (使用with调用JupyterTool时，with块结束后会自动清理资源)
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    tool = JupyterTool(concurrency=4)
    tool.set_tool_info()
    session_id = "test_session"
    codes = [
        "a = 10\nb = 20\na + b",
        "import time\ntime.sleep(2)\n'after sleep'",
        "for i in range(3):\n    print(i)",
        "1 / 0",  # This will raise an error
        "print('Hello, World!')",
        "time.sleep(130)",  # This will timeout
        "a * b",  # Should be 200 if state is preserved
    ]

    for code in codes:
        output = tool.execute_code(session_id, code)
        print(f"Code:\n{code}\nOutput:\n{output}\n{'-'*40}")

    # Clean up
    tool.del_client(session_id)
    tool.close()