"""
VLM 模型客户端

支持 Ollama 和 vLLM 两种模型服务
"""

import base64
from io import BytesIO

import requests
from PIL import Image


class VLMClient:
    """VLM 模型客户端（支持 Ollama 和 vLLM）"""

    def __init__(self, model_type: str, api_url: str, model_name: str):
        """
        Args:
            model_type: "ollama" 或 "vllm"
            api_url: API 地址，如 "http://localhost:11434" 或 "http://localhost:8000"
            model_name: 模型名称
        """
        self.model_type = model_type.lower()
        self.api_url = api_url.rstrip("/")
        self.model_name = model_name

        if self.model_type not in ["ollama", "vllm"]:
            raise ValueError(f"不支持的模型类型: {model_type}, 仅支持 'ollama' 或 'vllm'")

    def _image_to_base64(self, image: Image.Image) -> str:
        """将 PIL Image 转换为 base64 字符串"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64

    def query(self, image: Image.Image, prompt: str) -> str:
        """
        查询 VLM 模型

        Args:
            image: PIL Image 对象
            prompt: 文本提示

        Returns:
            模型响应文本
        """
        img_base64 = self._image_to_base64(image)

        if self.model_type == "ollama":
            return self._query_ollama(img_base64, prompt)
        elif self.model_type == "vllm":
            return self._query_vllm(img_base64, prompt)

    def _query_ollama(self, img_base64: str, prompt: str) -> str:
        """查询 Ollama API"""
        try:
            payload = {"model": self.model_name, "prompt": prompt, "images": [img_base64], "stream": False}

            response = requests.post(f"{self.api_url}/api/generate", json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"⚠ Ollama API 错误: {response.status_code}")
                return ""
        except Exception as e:
            print(f"⚠ Ollama 查询失败: {e}")
            return ""

    def _query_vllm(self, img_base64: str, prompt: str) -> str:
        """查询 vLLM API (OpenAI compatible)"""
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        ],
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.7,
            }

            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"⚠ vLLM API 错误: {response.status_code}")
                return ""
        except Exception as e:
            print(f"⚠ vLLM 查询失败: {e}")
            return ""
