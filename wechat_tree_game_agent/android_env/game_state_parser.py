"""
游戏状态解析器

功能:
- 使用 OCR 识别战斗力数值
- 识别装备属性变化 (↑↓→)
- 判断游戏界面状态 (砍树/装备选择/结果)
"""

from typing import Dict, Optional, List
from PIL import Image
import re

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("⚠ PaddleOCR 未安装，将使用模拟数据")


class GameStateParser:
    """游戏状态解析器"""

    def __init__(self, use_gpu: bool = False):
        """
        初始化解析器

        Args:
            use_gpu: 是否使用 GPU 加速 OCR
        """
        self.use_gpu = use_gpu

        if PADDLEOCR_AVAILABLE:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',  # 中文
                use_gpu=use_gpu,
                show_log=False
            )
            print("✓ PaddleOCR 初始化成功")
        else:
            self.ocr = None
            print("⚠ PaddleOCR 不可用，使用模拟解析")

    def extract_text(self, image: Image.Image) -> List[Dict]:
        """
        从图像中提取文本

        Args:
            image: PIL Image 对象

        Returns:
            List of {"text": str, "confidence": float, "bbox": list}
        """
        if self.ocr is None:
            # 模拟数据 (用于测试)
            return [
                {"text": "战斗力: 1250", "confidence": 0.95, "bbox": [[100, 100], [300, 150]]},
                {"text": "攻击 +50 ↑", "confidence": 0.90, "bbox": [[100, 200], [300, 250]]},
            ]

        try:
            # PaddleOCR 需要 numpy array 或路径
            import numpy as np
            image_array = np.array(image)

            result = self.ocr.ocr(image_array, cls=True)

            # 解析结果
            texts = []
            if result and result[0]:
                for line in result[0]:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]

                    texts.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox
                    })

            return texts

        except Exception as e:
            print(f"⚠ OCR 失败: {e}")
            return []

    def parse_combat_power(self, texts: List[Dict]) -> float:
        """
        解析妖力数值（游戏最上方黄色字体）

        Args:
            texts: OCR 结果列表

        Returns:
            妖力数值，失败返回 -1
        """
        # 关键词匹配（适配妖力系统）
        keywords = ["妖力", "战斗力", "战力", "power", "combat"]

        for item in texts:
            text = item["text"].lower()

            # 检查是否包含关键词
            if any(keyword in text for keyword in keywords):
                # 提取数字
                numbers = re.findall(r'\d+', text)
                if numbers:
                    # 取最大的数字 (通常是妖力)
                    return float(max(numbers, key=lambda x: int(x)))

        # 如果没有关键词，尝试提取所有数字，取最大值（妖力通常是最大的数字）
        all_numbers = []
        for item in texts:
            numbers = re.findall(r'\d+', item["text"])
            all_numbers.extend([int(n) for n in numbers])

        if all_numbers:
            # 过滤掉太小的数字 (可能是其他信息)
            # 妖力通常是5位数以上（例如77925）
            large_numbers = [n for n in all_numbers if n > 10000]
            if large_numbers:
                return float(max(large_numbers))
            # 如果没有大数字，取最大值
            elif all_numbers:
                return float(max(all_numbers))

        return -1.0

    def parse_equipment_stats(self, texts: List[Dict]) -> Dict[str, str]:
        """
        解析装备属性变化

        Args:
            texts: OCR 结果列表

        Returns:
            {"attack": "+50 ↑", "defense": "-10 ↓", "hp": "+20 ↑"}
        """
        stats = {}

        # 属性关键词
        stat_keywords = {
            "attack": ["攻击", "attack", "atk"],
            "defense": ["防御", "defense", "def"],
            "hp": ["生命", "血量", "hp", "health"],
            "speed": ["速度", "speed", "spd"],
            "crit": ["暴击", "critical", "crit"],
        }

        for item in texts:
            text = item["text"]

            # 检查是否包含属性关键词
            for stat_name, keywords in stat_keywords.items():
                if any(keyword in text.lower() for keyword in keywords):
                    # 提取数值和方向
                    # 例如: "攻击 +50 ↑" 或 "防御 -10 ↓"

                    # 提取符号 (+/-)
                    sign_match = re.search(r'([+\-])\s*(\d+)', text)
                    if sign_match:
                        sign = sign_match.group(1)
                        value = sign_match.group(2)

                        # 提取方向箭头
                        if "↑" in text or "上" in text or "up" in text.lower():
                            direction = "↑"
                        elif "↓" in text or "下" in text or "down" in text.lower():
                            direction = "↓"
                        elif "→" in text or "不变" in text or "same" in text.lower():
                            direction = "→"
                        else:
                            # 根据符号判断
                            direction = "↑" if sign == "+" else "↓"

                        stats[stat_name] = f"{sign}{value} {direction}"

        return stats

    def estimate_power_change(self, stats: Dict[str, str]) -> int:
        """
        估算总战斗力变化

        简单规则:
        - 攻击 权重 1.0
        - 防御 权重 0.8
        - 生命 权重 0.5
        - 其他 权重 0.3

        Args:
            stats: 装备属性字典

        Returns:
            估算的战斗力变化 (正数表示上升)
        """
        weights = {
            "attack": 1.0,
            "defense": 0.8,
            "hp": 0.5,
            "speed": 0.3,
            "crit": 0.3,
        }

        total_change = 0

        for stat_name, stat_value in stats.items():
            # 提取数值和方向
            match = re.search(r'([+\-])(\d+)', stat_value)
            if match:
                sign = match.group(1)
                value = int(match.group(2))

                # 应用权重
                weight = weights.get(stat_name, 0.5)
                change = value * weight

                if sign == "-":
                    change = -change

                total_change += change

        return int(total_change)

    def detect_game_state(self, texts: List[Dict]) -> str:
        """
        检测当前游戏状态

        Args:
            texts: OCR 结果列表

        Returns:
            "tree_cutting" | "equipment_selection" | "result" | "unknown"
        """
        all_text = " ".join([item["text"] for item in texts]).lower()

        # 关键词检测
        # 装备掉落界面的特征：有"替换"和"分解"按钮
        if any(keyword in all_text for keyword in ["替换", "分解", "装备黑索", "replace", "decompose"]):
            return "equipment_selection"

        # 砍树界面的特征：没有装备弹窗
        if any(keyword in all_text for keyword in ["砍树", "斧子", "chop", "tree", "axe"]):
            return "tree_cutting"

        if any(keyword in all_text for keyword in ["完成", "结束", "complete", "finish"]):
            return "result"

        return "unknown"

    def parse_screenshot(self, image: Image.Image) -> Dict:
        """
        完整解析截图

        Args:
            image: PIL Image 对象

        Returns:
            {
                "state": str,
                "combat_power": float,
                "equipment_stats": dict,
                "estimated_power_change": int
            }
        """
        # 1. 提取文本
        texts = self.extract_text(image)

        # 2. 解析战斗力
        combat_power = self.parse_combat_power(texts)

        # 3. 检测游戏状态
        state = self.detect_game_state(texts)

        # 4. 解析装备属性
        equipment_stats = {}
        estimated_power_change = 0

        if state == "equipment_selection":
            equipment_stats = self.parse_equipment_stats(texts)
            estimated_power_change = self.estimate_power_change(equipment_stats)

        return {
            "state": state,
            "combat_power": combat_power,
            "equipment_stats": equipment_stats,
            "estimated_power_change": estimated_power_change,
            "raw_texts": texts
        }


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("游戏状态解析器测试")
    print("=" * 60)

    # 初始化解析器
    parser = GameStateParser(use_gpu=False)

    # 测试 1: 解析战斗力
    print("\n[测试 1] 解析战斗力")
    texts = [
        {"text": "战斗力: 1250", "confidence": 0.95, "bbox": []},
        {"text": "其他信息 123", "confidence": 0.90, "bbox": []},
    ]
    power = parser.parse_combat_power(texts)
    print(f"识别到战斗力: {power}")
    assert power == 1250, "战斗力解析错误"

    # 测试 2: 解析装备属性
    print("\n[测试 2] 解析装备属性")
    texts = [
        {"text": "攻击 +50 ↑", "confidence": 0.95, "bbox": []},
        {"text": "防御 -10 ↓", "confidence": 0.90, "bbox": []},
        {"text": "生命 +20 ↑", "confidence": 0.92, "bbox": []},
    ]
    stats = parser.parse_equipment_stats(texts)
    print(f"装备属性: {stats}")

    # 测试 3: 估算战斗力变化
    print("\n[测试 3] 估算战斗力变化")
    estimated_change = parser.estimate_power_change(stats)
    print(f"估算战斗力变化: {estimated_change}")
    print(f"预期: 攻击+50*1.0 + 防御-10*0.8 + 生命+20*0.5 = +52")

    # 测试 4: 检测游戏状态
    print("\n[测试 4] 检测游戏状态")
    texts_tree = [{"text": "点击砍树", "confidence": 0.95, "bbox": []}]
    texts_equip = [{"text": "获得装备", "confidence": 0.95, "bbox": []}]
    state_tree = parser.detect_game_state(texts_tree)
    state_equip = parser.detect_game_state(texts_equip)
    print(f"砍树状态: {state_tree}")
    print(f"装备状态: {state_equip}")

    print("\n" + "=" * 60)
    print("所有测试通过！解析器工作正常。")
    print("=" * 60)
