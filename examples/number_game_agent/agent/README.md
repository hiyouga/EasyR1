# 执行步骤

## 1.在Android浏览器中打开游戏：
adb -s 101.43.137.83:5555 shell am start -a android.intent.action.VIEW -d "http://43.137.72.18:8000/number_game.html"
adb -s 192.144.164.28:5555 shell am start -a android.intent.action.VIEW -d "http://43.137.72.18:8000/number_game.html"

## 2. 确保设备已连接
adb connect 101.43.137.83:5555
adb connect 192.144.164.28:5555

## 3. 执行游戏脚本(带调试输出)(已测试无问题✅)
- ollama
```shell
cd examples
python number_game_agent/agent/number_game_play_agent.py \
    --model-type ollama \
    --api-url http://localhost:11434 \
    --model-name qwen2.5vl:3b \
    --devices 101.43.137.83:5555 \
    --debug
```
- vllm
```shell
cd examples
python number_game_agent/agent/number_game_play_agent.py \
    --model-type vllm \
    --api-url http://42.193.100.201:8000 \
    --model-name /home/ubuntu/EasyR1/checkpoints/Mobile_Number_Game_RL/quick_test_20251120_110841/global_step_35/actor/huggingface/ \
    --devices 101.43.137.83:5555 \
    --debug
```

## 6. 执行多局游戏
python number_game_agent/agent/number_game_play_agent.py \
--model-type ollama \
--api-url http://localhost:11434 \
--model-name qwen2.5vl:3b \
--devices 101.43.137.83:5555 \
--episodes 3 \
--debug

# 参数说明

- --model-type: 模型类型（ollama 或 vllm），默认 ollama
- --api-url: API地址，默认 http://localhost:11434
- --model-name: 模型名称，默认 qwen2.5vl:3b
- --devices: 设备列表，默认 101.43.137.83:5555
- --episodes: 运行局数，默认 1
- --debug: 开启调试模式，显示VLM输出

# 最简单的执行方式

直接运行（使用所有默认参数）：
cd /Users/zhangyuehua/Documents/my_fork/EasyR1
python number_game_agent/agent/number_game_play_agent.py --debug

游戏结束后，会提示你手动刷新浏览器，然后你可以再次运行脚本开始下一局。