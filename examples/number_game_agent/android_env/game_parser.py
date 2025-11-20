"""
数字选择游戏状态解析器

功能:
- 识别当前分数
- 识别3个数字卡片
- 识别当前回合
- 判断游戏状态
- 识别指示灯状态（规则）
"""

from typing import Dict, List, Tuple, Optional
from PIL import Image
import re

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("⚠ PaddleOCR 未安装，将使用模拟数据")


class NumberGameParser:
    """数字选择游戏解析器"""

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
                lang='ch',
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
                {"text": "100", "confidence": 0.98, "bbox": [[300, 100], [400, 150]]},
                {"text": "5", "confidence": 0.95, "bbox": [[100, 400], [200, 500]]},
                {"text": "3", "confidence": 0.96, "bbox": [[300, 400], [400, 500]]},
                {"text": "8", "confidence": 0.97, "bbox": [[500, 400], [600, 500]]},
            ]

        try:
            import numpy as np
            image_array = np.array(image)

            result = self.ocr.ocr(image_array, cls=True)

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

    def parse_score(self, texts: List[Dict]) -> int:
        """
        解析当前分数

        分数通常显示在屏幕上方，是一个较大的数字

        Args:
            texts: OCR 结果列表

        Returns:
            分数值，失败返回 -1
        """
        # 查找所有数字
        all_numbers = []

        for item in texts:
            text = item["text"]
            # 提取纯数字
            numbers = re.findall(r'\d+', text)

            for num in numbers:
                num_int = int(num)
                # 分数通常是 0-200 之间的两位或三位数
                if 0 <= num_int <= 300:
                    bbox = item["bbox"]
                    # 分数在屏幕上方（y 坐标小）
                    y_center = (bbox[0][1] + bbox[2][1]) / 2

                    all_numbers.append({
                        "value": num_int,
                        "y": y_center,
                        "confidence": item["confidence"]
                    })

        if all_numbers:
            # 选择 y 坐标最小（最上方）且最大的数字
            # 通常分数是屏幕最上方最显眼的大数字
            sorted_by_y = sorted(all_numbers, key=lambda x: x["y"])
            if sorted_by_y:
                return sorted_by_y[0]["value"]

        return -1

    def parse_card_numbers(self, texts: List[Dict], screen_width: int) -> List[int]:
        """
        解析 3 个卡片数字

        卡片数字通常在屏幕中央，横向排列，是1-9的单位数字

        Args:
            texts: OCR 结果列表
            screen_width: 屏幕宽度

        Returns:
            [左边数字, 中间数字, 右边数字]，失败返回空列表
        """
        card_candidates = []

        for item in texts:
            text = item["text"].strip()

            # 只考虑单个数字 (1-9)
            if text.isdigit() and len(text) == 1 and 1 <= int(text) <= 9:
                bbox = item["bbox"]

                # 计算中心点
                x_center = (bbox[0][0] + bbox[2][0]) / 2
                y_center = (bbox[0][1] + bbox[2][1]) / 2

                card_candidates.append({
                    "value": int(text),
                    "x": x_center,
                    "y": y_center,
                    "confidence": item["confidence"]
                })

        if len(card_candidates) < 3:
            return []

        # 过滤出 y 坐标相近的（同一行）
        # 卡片数字通常在屏幕中央或偏下
        sorted_by_y = sorted(card_candidates, key=lambda x: x["y"])

        # 取 y 坐标在中间范围的数字
        mid_y_candidates = []
        for i in range(len(sorted_by_y) - 2):
            group = sorted_by_y[i:i+3]
            y_diff = max(g["y"] for g in group) - min(g["y"] for g in group)

            # 如果 y 坐标差异小于屏幕高度的 10%，认为是同一行
            if y_diff < 100:  # 假设屏幕高度至少 1000px
                mid_y_candidates = group
                break

        if len(mid_y_candidates) < 3:
            # 退化方案：直接选择置信度最高的 3 个单位数字
            mid_y_candidates = sorted(card_candidates, key=lambda x: x["confidence"], reverse=True)[:3]

        if len(mid_y_candidates) < 3:
            return []

        # 按 x 坐标排序（从左到右）
        sorted_by_x = sorted(mid_y_candidates, key=lambda x: x["x"])

        return [card["value"] for card in sorted_by_x]

    def parse_round(self, texts: List[Dict]) -> int:
        """
        解析当前回合

        回合信息通常格式为 "回合: 1/10" 或 "1/10"

        Args:
            texts: OCR 结果列表

        Returns:
            当前回合数，失败返回 1
        """
        for item in texts:
            text = item["text"]

            # 匹配 "X/Y" 格式
            match = re.search(r'(\d+)\s*/\s*(\d+)', text)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))

                # 验证合理性
                if 1 <= current <= total <= 20:
                    return current

        return 1

    def detect_game_state(self, texts: List[Dict]) -> str:
        """
        检测游戏状态

        Args:
            texts: OCR 结果列表

        Returns:
            "playing" | "game_over" | "unknown"
        """
        all_text = " ".join([item["text"] for item in texts]).lower()

        # 检测游戏结束
        if any(keyword in all_text for keyword in ["游戏结束", "game over", "重新开始", "restart"]):
            return "game_over"

        # 检测正在游戏
        if any(keyword in all_text for keyword in ["当前分数", "选择", "数字", "回合"]):
            return "playing"

        return "unknown"

    def detect_indicator_color(self, image: Image.Image) -> Optional[str]:
        """
        通过颜色分析检测指示灯状态
        
        Args:
            image: PIL Image 对象
            
        Returns:
            "green" (选最大) | "red" (选最小) | "yellow" (选中间) | None
        """
        try:
            import numpy as np
            
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # 指示灯区域大约在屏幕上方 20%-35% 的位置
            indicator_region = img_array[int(height * 0.20):int(height * 0.38), :]
            
            # 计算主要颜色
            # 绿色：RGB(56, 239, 125) 附近
            # 红色：RGB(244, 92, 67) 附近  
            # 黄色：RGB(255, 210, 0) 附近
            
            green_mask = (
                (indicator_region[:, :, 0] < 100) &  # R 低
                (indicator_region[:, :, 1] > 150) &  # G 高
                (indicator_region[:, :, 2] > 80) &   # B 中等
                (indicator_region[:, :, 2] < 180)
            )
            
            red_mask = (
                (indicator_region[:, :, 0] > 200) &  # R 高
                (indicator_region[:, :, 1] < 150) &  # G 低
                (indicator_region[:, :, 2] < 150)    # B 低
            )
            
            yellow_mask = (
                (indicator_region[:, :, 0] > 200) &  # R 高
                (indicator_region[:, :, 1] > 150) &  # G 高
                (indicator_region[:, :, 2] < 100)    # B 低
            )
            
            green_pixels = np.sum(green_mask)
            red_pixels = np.sum(red_mask)
            yellow_pixels = np.sum(yellow_mask)
            
            # 选择像素数最多的颜色
            max_pixels = max(green_pixels, red_pixels, yellow_pixels)
            
            # 需要至少有一定数量的像素才认为是有效的
            threshold = 500  # 至少500个像素
            
            if max_pixels < threshold:
                return None
                
            if green_pixels == max_pixels:
                return "green"
            elif red_pixels == max_pixels:
                return "red"
            elif yellow_pixels == max_pixels:
                return "yellow"
                
        except Exception as e:
            print(f"⚠ 指示灯颜色检测失败: {e}")
            
        return None

    def parse_rule_from_indicator(self, indicator_color: Optional[str]) -> Optional[str]:
        """
        从指示灯颜色解析游戏规则
        
        Args:
            indicator_color: "green" | "red" | "yellow" | None
            
        Returns:
            "max" | "min" | "mid" | None
        """
        if indicator_color == "green":
            return "max"
        elif indicator_color == "red":
            return "min"
        elif indicator_color == "yellow":
            return "mid"
        return None

    def parse_screenshot(self, image: Image.Image, screen_width: int = 1080) -> Dict:
        """
        完整解析截图

        Args:
            image: PIL Image 对象
            screen_width: 屏幕宽度（用于计算卡片位置）

        Returns:
            {
                "state": str,
                "score": int,
                "numbers": [int, int, int],
                "round": int,
                "rule": str,  # "max" | "min" | "mid"
                "indicator_color": str,  # "green" | "red" | "yellow"
                "raw_texts": list
            }
        """
        # 1. 提取文本
        texts = self.extract_text(image)

        # 2. 解析分数
        score = self.parse_score(texts)

        # 3. 解析卡片数字
        numbers = self.parse_card_numbers(texts, screen_width)

        # 4. 解析回合
        round_num = self.parse_round(texts)

        # 5. 检测游戏状态
        state = self.detect_game_state(texts)

        # 6. 检测指示灯颜色和规则
        indicator_color = self.detect_indicator_color(image)
        rule = self.parse_rule_from_indicator(indicator_color)

        return {
            "state": state,
            "score": score,
            "numbers": numbers,
            "round": round_num,
            "rule": rule,
            "indicator_color": indicator_color,
            "raw_texts": texts
        }


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("数字选择游戏解析器测试")
    print("=" * 60)

    # 初始化解析器
    parser = NumberGameParser(use_gpu=False)

    # 测试 1: 解析分数
    print("\n[测试 1] 解析分数")
    texts = [
        {"text": "当前分数", "confidence": 0.95, "bbox": [[100, 50], [200, 80]]},
        {"text": "100", "confidence": 0.98, "bbox": [[150, 90], [250, 140]]},
        {"text": "5", "confidence": 0.95, "bbox": [[100, 400], [200, 500]]},
    ]
    score = parser.parse_score(texts)
    print(f"识别到分数: {score}")
    assert score == 100, "分数解析错误"

    # 测试 2: 解析卡片数字
    print("\n[测试 2] 解析卡片数字")
    texts = [
        {"text": "5", "confidence": 0.95, "bbox": [[100, 400], [200, 500]]},
        {"text": "3", "confidence": 0.96, "bbox": [[300, 405], [400, 505]]},
        {"text": "8", "confidence": 0.97, "bbox": [[500, 398], [600, 498]]},
    ]
    numbers = parser.parse_card_numbers(texts, screen_width=1080)
    print(f"识别到数字: {numbers}")
    assert numbers == [5, 3, 8], "卡片数字解析错误"

    # 测试 3: 解析回合
    print("\n[测试 3] 解析回合")
    texts = [
        {"text": "回合: 3/10", "confidence": 0.95, "bbox": [[100, 150], [200, 180]]},
    ]
    round_num = parser.parse_round(texts)
    print(f"识别到回合: {round_num}")
    assert round_num == 3, "回合解析错误"

    print("\n" + "=" * 60)
    print("所有测试通过！解析器工作正常。")
    print("=" * 60)
