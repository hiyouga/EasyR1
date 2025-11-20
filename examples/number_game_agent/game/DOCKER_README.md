# 数字选择游戏 - Docker 镜像

## 📦 镜像信息

**镜像名称**: `number-game-rl`  
**版本**: `v1.0` / `latest`  
**架构**: `linux/amd64`  
**大小**: ~124MB  
**基础镜像**: `python:3.11-slim`

## 🚀 使用方法

### 1. 推送到仓库

```bash
# 方式1: 推送到 Docker Hub
docker tag number-game-rl:v1.0 YOUR_USERNAME/number-game-rl:v1.0
docker tag number-game-rl:latest YOUR_USERNAME/number-game-rl:latest
docker push YOUR_USERNAME/number-game-rl:v1.0
docker push YOUR_USERNAME/number-game-rl:latest

# 方式2: 推送到私有仓库
docker tag number-game-rl:v1.0 registry.example.com/number-game-rl:v1.0
docker tag number-game-rl:latest registry.example.com/number-game-rl:latest
docker push registry.example.com/number-game-rl:v1.0
docker push registry.example.com/number-game-rl:latest
```

### 2. 运行容器

```bash
# 运行游戏服务器（端口8000）
docker run -d \
  --name number-game \
  -p 8000:8000 \
  number-game-rl:v1.0

# 自定义端口（例如映射到9000）
docker run -d \
  --name number-game \
  -p 9000:8000 \
  number-game-rl:v1.0
```

### 3. 访问游戏

打开浏览器访问：
```
http://localhost:8000/number_game.html
```

### 4. Kubernetes 部署

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: number-game
spec:
  containers:
  - name: game
    image: YOUR_REGISTRY/number-game-rl:v1.0
    ports:
    - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: number-game-svc
spec:
  type: NodePort
  ports:
  - port: 8000
    targetPort: 8000
    nodePort: 30000
  selector:
    app: number-game
```

## 🎮 游戏说明

这是一个**条件反转数字选择游戏**，用于强化学习训练。

### 游戏规则

1. **观察指示灯**（屏幕上方3个圆形）：
   - 🟢 绿灯亮：选择**最大**的数字
   - 🔴 红灯亮：选择**最小**的数字
   - 🟡 黄灯亮：选择**中间**的数字

2. **得分规则**：
   - 选对：+10 分
   - 选错：-10 分

3. **游戏目标**：完成10轮，获得最高分

### 适配分辨率

- 优化适配：720x1280（Android设备）
- 兼容：桌面浏览器、平板、手机

## 🔧 镜像内容

```
/app/
  └── number_game.html  # 游戏HTML文件（包含CSS和JavaScript）
```

## 📝 环境变量

无需配置环境变量，开箱即用。

## 🐛 故障排查

### 容器无法启动
```bash
docker logs number-game
```

### 端口冲突
```bash
# 更换端口
docker run -d --name number-game -p 9000:8000 number-game-rl:v1.0
```

### 查看容器状态
```bash
docker ps -a | grep number-game
```

## 📄 许可证

本项目用于强化学习研究和教学目的。
