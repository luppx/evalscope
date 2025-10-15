import requests
import threading
from evalscope.api.tool import ToolInfo, ToolParams
from evalscope.utils.logger import get_logger

logger = get_logger()

class PythonSandbox:
    def __init__(self):
        # Configuration for tool execution
        self.tool_configs = {
            "max_turns": 16,
            "max_tool_calls": 16,
            "tool_concurrency": 32,  # Aggressive: 32 concurrent processes
            # Python interpreter settings
            "sandbox_url": "http://klb-dgx-010:15433/run_code",
            "python_timeout": 120,  # 2 minutes for complex calculations
        }

        self.sandbox_url = self.tool_configs["sandbox_url"]
        self.code_run_timeout = self.tool_configs["python_timeout"]

        # Use a threading.Semaphore since execution and HTTP client are synchronous (requests)
        self.semaphore = threading.Semaphore(self.tool_configs["tool_concurrency"])
        self.set_tool_info()

    def set_tool_info(self):
        self.tool_info = ToolInfo(
            name="python",
            description="Executes Python code in a stateless sandbox. The code must be a complete script with all necessary imports, and outputs in the code script must explicitly call the print() function.",
            parameters=ToolParams(
                properties={
                    "code": {
                        "type": "string",
                        "description": 'The Python script to execute. It must include all required imports and use print() function to display any results.',
                    },
                },
                required=["code"],
            ),
        )
    
    def get_tool_info(self) -> ToolInfo:
        return self.tool_info

    def call_code_sandbox(self, script: str, language: str = "python") -> str:
        """
        Call our internal code sandbox to execute the script by http request.
        """
        headers = {"Content-Type": "application/json", 'Accept': 'application/json'}
        data = {
            "code": script,
            "language": language,
            "run_timeout": self.code_run_timeout,
        }

        try:
            response = requests.post(url=self.sandbox_url, headers=headers, json=data, 
                                    proxies={"http": None, "https": None})
            logger.info(f"Code sandbox request: {data}, response: {response.text}")
            if response.status_code == 200:
                rsp_json = response.json()
                run_result = rsp_json.get("run_result", {})
                status = run_result.get("status", "")
                stdout = run_result.get("stdout", "").replace("User customization module loaded!\n", "", 1)
                stderr = run_result.get("stderr", "")
                if status == "TimeLimitExceeded":
                    return f"TimeLimitExceeded Error. Code execution timed out, with a timeout limit of {self.code_run_timeout} seconds."
                elif stderr:
                    return stderr
                else:
                    return stdout
            else:
                return f"Error occurred when calling code sandbox. HTTP response code: {response.status_code}, HTTP response: {response.text}"
        except Exception as e:
            return f"Exception occurred when calling code sandbox: {e}"

    def execute_code(self, code: str) -> str:
        # Safely limit concurrency; ensure release even if an exception occurs
        with self.semaphore:
            return self.call_code_sandbox(code, language="python")