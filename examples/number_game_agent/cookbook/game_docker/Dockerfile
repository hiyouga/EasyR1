# 数字选择游戏 - Docker镜像
# 平台: linux/amd64
FROM --platform=linux/amd64 python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制游戏文件
COPY number_game.html /app/

# 暴露端口
EXPOSE 8000

# 启动HTTP服务器
CMD ["python", "-m", "http.server", "8000"]
