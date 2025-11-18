# 微信砍树游戏训练数据集

## 📊 数据集概览

- **总计**: 36 条训练数据（29 训练 + 7 验证）
- **数据来源**: 用户提供的 40 张游戏截图
- **生成时间**: 2025-11-18
- **数据格式**: JSONL (JSON Lines)

---

## 📁 文件结构

```
wechat_tree_game_agent/data/
├── annotations.json          # 自动生成的标注文件（中间产物）
├── tree_game_train.jsonl     # 训练集（29条）
├── tree_game_val.jsonl       # 验证集（7条）
├── auto_annotate.py          # 自动标注脚本
├── build_dataset.py          # 数据集构建脚本
└── DATASET_SUMMARY.md        # 本文档
```

原始截图目录: `/Users/zhangyuehua/Desktop/tree_cutting/`

---

## 📈 数据分布

### 游戏状态分布

| 状态 | 数量 | 占比 | 说明 |
|------|------|------|------|
| `equipment_selection` | 27 | 75% | 装备掉落界面 |
| `tree_cutting` | 9 | 25% | 砍树主界面 |

### 动作类型分布

| 动作 | 数量 | 占比 | 说明 |
|------|------|------|------|
| `replace()` | 22 | 61% | 替换装备（妖力上升） |
| `click(180, 1000)` | 9 | 25% | 点击砍树按钮 |
| `decompose()` | 5 | 14% | 分解装备（妖力下降） |

### 装备决策分布

| 决策类型 | 数量 | 说明 |
|----------|------|------|
| **正向决策**（妖力↑） | 20 | 装备属性整体提升 |
| **负向决策**（妖力↓） | 5 | 装备属性整体下降 |
| **混合决策**（需计算） | 2 | 属性有升有降，需要权衡 |

---

## 🎯 数据样例

### 样例 1: 砍树动作

```json
{
  "prompt": "当前是砍树界面，请点击屏幕下方中央的斧子开始砍树。",
  "images": ["/Users/zhangyuehua/Desktop/tree_cutting/主界面_001.png"],
  "answer": "<action>click(180, 1000)</action>",
  "state": "tree_cutting",
  "description": "点击屏幕下方中央的斧子按钮进行砍树"
}
```

### 样例 2: 替换装备（妖力上升）

```json
{
  "prompt": "装备掉落了！新装备属性整体提升，妖力会上升。",
  "images": ["/Users/zhangyuehua/Desktop/tree_cutting/装备掉落_001.png"],
  "answer": "<action>replace()</action>",
  "state": "equipment_selection",
  "description": "装备属性整体提升，妖力上升，应该替换",
  "expected_change": "positive"
}
```

### 样例 3: 分解装备（妖力下降）

```json
{
  "prompt": "装备掉落了！新装备属性整体下降，妖力会下降。",
  "images": ["/Users/zhangyuehua/Desktop/tree_cutting/装备掉落_降低战力_001.png"],
  "answer": "<action>decompose()</action>",
  "state": "equipment_selection",
  "description": "装备属性整体下降，妖力下降，应该分解",
  "expected_change": "negative"
}
```

---

## 🔧 数据生成流程

### Step 1: 自动分类和标注

```bash
python wechat_tree_game_agent/data/auto_annotate.py
```

**输出**: `annotations.json` (40 条标注)

**分类规则**:
- 文件名包含"主界面"或"砍树的位置" → `tree_cutting`
- 文件名包含"装备掉落" + "降低战力" → `equipment_negative`
- 文件名包含"装备掉落" + "混合" → `equipment_mixed`
- 文件名包含"装备掉落" → `equipment_positive`
- 文件名包含"装备替换结果" → `result` (验证用)

### Step 2: 构建训练数据集

```bash
python wechat_tree_game_agent/data/build_dataset.py
```

**输出**:
- `tree_game_train.jsonl` (29 条)
- `tree_game_val.jsonl` (7 条)

**转换规则**:
1. 将标注转换为 EasyR1 JSONL 格式
2. 添加 prompt 和 answer
3. 随机划分训练/验证集（8:2比例）
4. 排除结果界面（用于人工验证）

---

## 📋 数据字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt` | string | ✅ | 给模型的任务描述 |
| `images` | array | ✅ | 图片路径列表（通常1张） |
| `answer` | string | ✅ | 模型应输出的标准答案（格式:`<action>...</action>`） |
| `state` | string | ✅ | 游戏状态（`tree_cutting` / `equipment_selection`） |
| `description` | string | ✅ | 人类可读的决策解释 |
| `expected_change` | string | ⚠️ | 妖力预期变化（`positive` / `negative` / `mixed`），仅装备界面有 |

---

## 🎮 游戏规则映射

### 妖力系统

```
妖力 = 攻击 + 生命 + 防御 + 敏捷 + 闪避 + 其他属性
```

**训练目标**: 最大化妖力

### 奖励函数

| 妖力变化 | 奖励 | 说明 |
|----------|------|------|
| 妖力上升 | +1.0 ~ +2.0 | 正确选择"替换" |
| 妖力不变 | 0.0 | 正确选择"分解" |
| 妖力下降 | -1.0 ~ -2.0 | 错误选择"替换" |

---

## ✅ 数据质量检查

### 已验证项

- ✅ 所有图片路径存在且可访问
- ✅ 所有动作格式符合 `<action>...</action>` 规范
- ✅ 状态标签正确（通过人工抽查）
- ✅ 训练/验证集无重复数据
- ✅ 数据集覆盖所有3种动作类型

### 需要注意的项

⚠️ **混合决策样本较少** (仅2条)
- 建议在实际训练中关注这类样本的学习效果
- 如果模型在复杂属性组合上表现不佳，需要补充更多混合样本

⚠️ **负向决策样本较少** (仅5条)
- 可能导致模型倾向于总是选择"替换"
- 建议在训练时监控 precision/recall 指标

---

## 🚀 下一步操作

1. **验证数据集**
   ```bash
   # 查看训练集前3条
   head -3 wechat_tree_game_agent/data/tree_game_train.jsonl

   # 查看验证集
   cat wechat_tree_game_agent/data/tree_game_val.jsonl
   ```

2. **更新训练配置**
   ```yaml
   # wechat_tree_game_agent/config/tree_game_grpo.yaml
   data:
     train_files: wechat_tree_game_agent/data/tree_game_train.jsonl
     val_files: wechat_tree_game_agent/data/tree_game_val.jsonl
   ```

3. **开始训练**
   ```bash
   bash wechat_tree_game_agent/scripts/train.sh
   ```

---

## 📝 数据集版本

- **v1.0** (2025-11-18)
  - 初始版本
  - 36条训练数据
  - 基于40张手动收集的截图
  - 支持3种动作类型（砍树、替换、分解）

---

## 🔄 数据扩充建议

如果需要提升模型性能，建议补充：

1. **更多负向样本** (目标: 10-15条)
   - 各种属性大幅下降的装备
   - 确保模型学会正确"分解"

2. **更多混合样本** (目标: 10-15条)
   - 攻击↑ + 防御↓↓ (应该分解)
   - 攻击↑↑ + 防御↓ (应该替换)
   - 训练模型的权衡能力

3. **边界案例** (目标: 5-10条)
   - 妖力几乎不变的装备
   - 微弱提升/下降的装备
   - 测试模型的细粒度判断

---

**生成时间**: 2025-11-18
**最后更新**: 2025-11-18
**维护者**: Claude Code
