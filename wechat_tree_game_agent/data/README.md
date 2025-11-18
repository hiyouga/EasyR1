# 数据集使用指南

本目录包含微信砍树游戏的训练数据集和相关工具。

---

## 🎯 快速开始

### 1. 查看数据集统计

```bash
# 查看数据集摘要
cat wechat_tree_game_agent/data/DATASET_SUMMARY.md

# 统计训练集数量
wc -l wechat_tree_game_agent/data/tree_game_train.jsonl
# 输出: 29 条

# 统计验证集数量
wc -l wechat_tree_game_agent/data/tree_game_val.jsonl
# 输出: 7 条
```

### 2. 查看数据样例

```bash
# 查看训练集第一条
head -1 wechat_tree_game_agent/data/tree_game_train.jsonl | python -m json.tool

# 查看所有砍树动作
grep "click" wechat_tree_game_agent/data/tree_game_train.jsonl

# 查看所有替换动作
grep "replace" wechat_tree_game_agent/data/tree_game_train.jsonl

# 查看所有分解动作
grep "decompose" wechat_tree_game_agent/data/tree_game_train.jsonl
```

### 3. 验证数据质量

```bash
# 检查所有图片是否存在
python -c "
import json
from pathlib import Path

with open('wechat_tree_game_agent/data/tree_game_train.jsonl') as f:
    for line in f:
        data = json.loads(line)
        for img in data['images']:
            if not Path(img).exists():
                print(f'Missing: {img}')
"

# 如果没有输出，说明所有图片都存在 ✓
```

---

## 📦 已生成的文件

| 文件 | 说明 | 大小 |
|------|------|------|
| `tree_game_train.jsonl` | 训练集 | 29 条 |
| `tree_game_val.jsonl` | 验证集 | 7 条 |
| `annotations.json` | 中间标注文件 | 40 条 |
| `DATASET_SUMMARY.md` | 数据集详细文档 | - |
| `README.md` | 本文档 | - |

---

## 🔧 工具脚本

### `auto_annotate.py` - 自动标注工具

**用途**: 根据截图文件名自动生成标注

```bash
python wechat_tree_game_agent/data/auto_annotate.py
```

**输入**: `/Users/zhangyuehua/Desktop/tree_cutting/*.png`
**输出**: `annotations.json`

**支持的文件名模式**:
- `主界面_*.png` → 砍树动作
- `砍树的位置_*.png` → 砍树动作
- `装备掉落_*.png` → 替换动作（妖力上升）
- `装备掉落_降低战力_*.png` → 分解动作（妖力下降）
- `装备掉落_混合_*.png` → 需要手动检查
- `装备替换结果_*.png` → 跳过（验证用）

---

### `build_dataset.py` - 数据集构建工具

**用途**: 将标注转换为 EasyR1 训练格式

```bash
python wechat_tree_game_agent/data/build_dataset.py \
    --input wechat_tree_game_agent/data/annotations.json \
    --screenshot-dir /Users/zhangyuehua/Desktop/tree_cutting \
    --output wechat_tree_game_agent/data/ \
    --val-ratio 0.2
```

**参数说明**:
- `--input`: 标注文件路径
- `--screenshot-dir`: 截图目录
- `--output`: 输出目录
- `--val-ratio`: 验证集比例（默认 0.2）

**输出**:
- `tree_game_train.jsonl` (80%)
- `tree_game_val.jsonl` (20%)

---

## 📊 数据格式

### JSONL 格式 (训练数据)

```json
{
  "prompt": "当前是砍树界面，请点击屏幕下方中央的斧子开始砍树。",
  "images": ["/path/to/screenshot.png"],
  "answer": "<action>click(180, 1000)</action>",
  "state": "tree_cutting",
  "description": "点击屏幕下方中央的斧子按钮进行砍树"
}
```

**必需字段**:
- `prompt`: 任务描述
- `images`: 图片路径列表
- `answer`: 标准答案（格式: `<action>...</action>`）

**可选字段**:
- `state`: 游戏状态
- `description`: 决策说明
- `expected_change`: 妖力变化（装备界面）

---

## 🎮 支持的动作类型

### 1. 点击砍树

```json
{
  "answer": "<action>click(180, 1000)</action>",
  "description": "点击屏幕下方中央的斧子"
}
```

**使用场景**: 主界面/砍树界面

### 2. 替换装备

```json
{
  "answer": "<action>replace()</action>",
  "description": "妖力上升，应该替换"
}
```

**使用场景**: 装备掉落，妖力会上升

### 3. 分解装备

```json
{
  "answer": "<action>decompose()</action>",
  "description": "妖力下降，应该分解"
}
```

**使用场景**: 装备掉落，妖力会下降

---

## 🔄 重新生成数据集

如果你添加了新的截图或修改了标注，可以重新生成数据集：

```bash
# Step 1: 重新自动标注
python wechat_tree_game_agent/data/auto_annotate.py

# Step 2: 检查并手动修正 annotations.json（如果需要）
# 使用文本编辑器打开 annotations.json

# Step 3: 重新构建数据集
python wechat_tree_game_agent/data/build_dataset.py

# Step 4: 验证新数据集
head -3 wechat_tree_game_agent/data/tree_game_train.jsonl | python -m json.tool
```

---

## 📈 数据集扩充指南

### 当前数据分布

- ✅ 砍树动作: 9 条 (充足)
- ✅ 替换动作: 22 条 (充足)
- ⚠️ 分解动作: 5 条 (偏少)
- ⚠️ 混合决策: 2 条 (偏少)

### 建议补充

**优先级 1: 分解动作样本**
- 目标: 增加到 10-15 条
- 方法: 收集更多"妖力明显下降"的装备截图

**优先级 2: 混合决策样本**
- 目标: 增加到 10-15 条
- 方法: 收集"属性有升有降"的复杂装备

**优先级 3: 边界案例**
- 目标: 5-10 条
- 方法: 收集妖力变化很小的装备（+10, -5 等）

### 添加新数据的步骤

1. 将新截图放入 `/Users/zhangyuehua/Desktop/tree_cutting/`
2. 按照命名规范命名文件（例如: `装备掉落_降低战力_005.png`）
3. 运行 `auto_annotate.py` 重新生成标注
4. 运行 `build_dataset.py` 重新生成数据集

---

## ❓ 常见问题

### Q1: 如何手动修改标注？

编辑 `annotations.json`，修改对应条目的 `action` 字段：

```json
{
  "id": 10,
  "image": "装备掉落_混合_001.png",
  "state": "equipment_selection",
  "action": "decompose()",  // ← 修改这里
  "description": "虽然攻击上升，但防御下降更多，总妖力下降"
}
```

然后重新运行 `build_dataset.py`。

### Q2: 图片路径错误怎么办？

确认截图确实在指定目录：

```bash
ls /Users/zhangyuehua/Desktop/tree_cutting/
```

如果截图移动了位置，修改 `build_dataset.py` 的 `--screenshot-dir` 参数。

### Q3: 如何调整训练/验证集比例？

修改 `build_dataset.py` 的 `--val-ratio` 参数：

```bash
# 10% 验证集
python wechat_tree_game_agent/data/build_dataset.py --val-ratio 0.1

# 30% 验证集
python wechat_tree_game_agent/data/build_dataset.py --val-ratio 0.3
```

---

## 📞 技术支持

- 查看数据集详细信息: `cat DATASET_SUMMARY.md`
- 查看项目文档: `cat ../README.md`
- 报告问题: 在项目 GitHub Issues 提交

---

**最后更新**: 2025-11-18
