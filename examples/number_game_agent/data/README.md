# 1. 在Android浏览器中打开游戏：
```shell
adb -s 101.43.137.83:5555 shell am start -a android.intent.action.VIEW -d "http://43.137.72.18:8000/number_game.html"
adb -s 192.144.164.28:5555 shell am start -a android.intent.action.VIEW -d "http://43.137.72.18:8000/number_game.html"
adb -s 49.233.170.121:5555 shell am start -a android.intent.action.VIEW -d "http://43.137.72.18:8000/number_game.html"
```

# 2. 收集训练数据
```shell
python examples/number_game_agent/data/collect_game_data.py \
    --devices 101.43.137.83:5555 192.144.164.28:5555 49.233.170.121:5555 \
    --episodes 1 \
    --parallel \
    --max-workers 3 \
    --output-dir game_data_raw
```


# 3. 标注数据并生成训练数据集
```shell
python examples/number_game_agent/data/annotate_game_data.py \
    --data-dir examples/number_game_agent/data/game_data_raw \
    --output-dir examples/number_game_agent/data/game_dataset_1119_v2 \
    --train-ratio 0.8 \
    --vlm-model qwen2.5vl:7b \
    --debug
```
