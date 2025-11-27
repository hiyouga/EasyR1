# 连接 android 设备玩游戏步骤

android 云端 android 设备创建可参考: https://github.com/tkestack/tke-ai-playbook/pull/20

## 1.在Android浏览器中打开游戏：
adb -s <android_ip>:5555 shell am start -a android.intent.action.VIEW -d "http://<game_ip>:8000/number_game.html"

## 2. 确保设备已连接
adb connect <android_ip>:5555

## 3. 执行游戏脚本
- ollama
```shell
python examples/android_gui_cookbook/play_agent.py \
    --model-type ollama \
    --api-url http://localhost:11434 \
    --model-name qwen2.5vl:3b \
    --devices <android_ip>:5555 \
    --debug
```
- vllm
```shell
python examples/android_gui_cookbook/play_agent.py \
    --model-type vllm \
    --api-url <vllm_ip> \
    --model-name <model_id> \
    --devices <android_ip>:5555 \
    --debug
```

# 参数说明

- --model-type: 模型类型（ollama 或 vllm），默认 ollama
- --api-url: API地址，默认 http://localhost:11434
- --model-name: 模型名称，默认 qwen2.5vl:3b
- --devices: 设备列表
- --episodes: 运行局数，默认 1
- --debug: 开启调试模式，显示VLM输出
