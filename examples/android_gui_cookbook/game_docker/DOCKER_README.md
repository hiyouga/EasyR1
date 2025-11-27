# 数字选择游戏 - Docker 镜像

## 📦 镜像信息

**镜像名称**: `number-game-rl`  
**当前版本**: `v1.4`  
**镜像仓库**: `ccr.ccs.tencentyun.com/yuehuazhang/number-game-rl`  
**架构**: `linux/amd64`  
**大小**: ~124MB
**基础镜像**: `python:3.11-slim`

## 🚀 使用方法

### 1. Docker 部署

```bash
# 从腾讯云镜像仓库拉取并运行
docker run -d \
  --name number-game \
  -p 8000:8000 \
  ccr.ccs.tencentyun.com/yuehuazhang/number-game-rl:v1.4

# 自定义端口（例如映射到9000）
docker run -d \
  --name number-game \
  -p 9000:8000 \
  ccr.ccs.tencentyun.com/yuehuazhang/number-game-rl:v1.4
```

### 2. 访问游戏

打开浏览器访问：
```
http://localhost:8000/number_game.html
```

### 3. Kubernetes 部署（推荐）

使用提供的 `game.yaml` 配置文件进行部署：

```bash
# 部署到 Kubernetes 集群
kubectl apply -f game.yaml
```

**game.yaml 配置说明：**

```yaml
# Deployment 配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: number-game
spec:
  replicas: 1                                    # 副本数
  template:
    spec:
      containers:
      - name: number-game
        image: ccr.ccs.tencentyun.com/yuehuazhang/number-game-rl:v1.4
        imagePullPolicy: IfNotPresent            # 镜像拉取策略
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "2"                             # CPU限制：2核
            memory: 4Gi                          # 内存限制：4GB
          requests:
            cpu: "2"                             # CPU请求：2核
            memory: 4Gi                          # 内存请求：4GB

---
# Service 配置（LoadBalancer类型）
apiVersion: v1
kind: Service
metadata:
  name: number-game
  annotations:
    service.cloud.tencent.com/direct-access: "true"  # 腾讯云直连
spec:
  type: LoadBalancer                             # 使用负载均衡器
  allocateLoadBalancerNodePorts: false           # 不分配节点端口
  ports:
  - name: 8000-8000-tcp
    port: 8000
    targetPort: 8000
    protocol: TCP
  selector:
    k8s-app: number-game
```

**部署后访问：**

```bash
# 查看服务状态
kubectl get svc number-game

# 获取 LoadBalancer 外部IP
kubectl get svc number-game -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# 访问游戏（替换为实际的外部IP）
# http://<EXTERNAL-IP>:8000/number_game.html
```

**扩缩容：**

```bash
# 扩展副本数
kubectl scale deployment number-game --replicas=3

# 查看 Pod 状态
kubectl get pods -l k8s-app=number-game
```

**删除部署：**

```bash
kubectl delete -f game.yaml
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
