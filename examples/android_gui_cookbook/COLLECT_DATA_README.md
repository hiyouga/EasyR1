# 连接 android 设备收集训练数据步骤

# 1. 在 Android 中打开游戏
```shell
adb -s <android_ip>:5555 shell am start -a android.intent.action.VIEW -d "http://<game_ip>:8000/number_game.html"
```

# 2. 收集训练数据
max-workers: number of devices

```shell
python examples/android_gui_cookbook/collect_data.py \
    --devices <android_ip1>:5555 <android_ip2>:5555 <android_ip3>:5555 \
    --episodes 1 \
    --parallel \
    --max-workers 3 \
    --output-dir game_data_raw
```
