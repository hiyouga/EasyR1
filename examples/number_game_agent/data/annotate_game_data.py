"""
数据标注脚本 - 使用VLM批量标注收集的游戏截图

功能:
1. 从question.png中提取：灯光颜色 + 3个数字
2. 从result.png中提取：正确答案的index
3. 生成problem文本（包含具体的灯光和数字信息）
4. 输出训练集和测试集JSONL文件（符合EasyR1格式）
"""

import argparse
import base64
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image
from tqdm import tqdm

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("⚠ PaddleOCR未安装，将使用VLM提取数字（准确率较低）")
    print("  安装命令: pip install paddleocr")


class GameDataAnnotator:
    """游戏数据标注器"""
    
    def __init__(
        self,
        vlm_base_url: str = "http://localhost:11434",
        vlm_model: str = "qwen2.5-vl:latest",
        use_ocr: bool = True,
        debug: bool = False
    ):
        self.vlm_base_url = vlm_base_url
        self.vlm_model = vlm_model
        self.debug = debug
        self.use_ocr = use_ocr and PADDLEOCR_AVAILABLE
        
        # 初始化OCR
        if self.use_ocr:
            try:
                print("初始化 PaddleOCR...")
                self.ocr = PaddleOCR(lang='en')
                print("✓ PaddleOCR 已就绪")
            except Exception as e:
                print(f"⚠ PaddleOCR初始化失败: {e}")
                print("  降级使用VLM提取数字")
                self.ocr = None
                self.use_ocr = False
        else:
            self.ocr = None
            if use_ocr:
                print("⚠ 使用VLM提取数字（建议安装PaddleOCR以提高准确率）")
        
        print(f"VLM配置: {vlm_base_url} - {vlm_model}")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def call_vlm(self, image_path: str, prompt: str) -> str:
        """
        调用Ollama VLM进行图像分析
        
        Args:
            image_path: 图片路径
            prompt: 提示词
            
        Returns:
            VLM的响应文本
        """
        try:
            # 编码图片
            image_base64 = self.encode_image_to_base64(image_path)
            
            # 构建请求
            url = f"{self.vlm_base_url}/api/generate"
            payload = {
                "model": self.vlm_model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }
            
            # 发送请求
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            print(f"⚠ VLM调用失败: {e}")
            return ""
    
    def extract_numbers_with_ocr(self, image_path: str) -> Optional[List[int]]:
        """
        使用OCR从截图中提取3个数字
        
        Returns:
            [left_number, middle_number, right_number] 或 None
        """
        try:
            # 使用新版PaddleOCR的predict API
            result = self.ocr.predict(image_path)
            
            if not result or len(result) == 0:
                return None
            
            # 新版API返回字典列表
            ocr_result = result[0] if isinstance(result, list) else result
            
            # 提取识别到的文本和位置
            rec_texts = ocr_result.get('rec_texts', [])
            rec_boxes = ocr_result.get('rec_boxes', [])
            rec_scores = ocr_result.get('rec_scores', [])
            
            if not rec_texts or len(rec_texts) == 0:
                return None
            
            # 提取所有数字及其位置
            detections = []
            for i, text in enumerate(rec_texts):
                # 只保留纯数字
                if text.isdigit() and rec_scores[i] > 0.5:
                    box = rec_boxes[i]
                    # box格式: [x_min, y_min, x_max, y_max]
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    
                    detections.append({
                        'text': int(text),
                        'x': center_x,
                        'y': center_y,
                        'confidence': rec_scores[i]
                    })
            
            if self.debug:
                print(f"  OCR检测到 {len(detections)} 个数字: {[d['text'] for d in detections]}")
            
            if len(detections) < 3:
                if self.debug:
                    print(f"  OCR数字不足3个，降级使用VLM")
                return None
            
            # 过滤：保留y坐标在合理范围内的数字（卡片区域约在y>600）
            card_detections = [d for d in detections if d['y'] > 600]
            
            if len(card_detections) < 3:
                # 如果过滤后不够，使用所有检测
                card_detections = detections
            
            # 按x坐标排序（从左到右）
            card_detections.sort(key=lambda d: d['x'])
            
            # 如果检测到多于3个数字，取置信度最高的3个
            if len(card_detections) > 3:
                # 先按置信度排序取前3
                card_detections.sort(key=lambda d: d['confidence'], reverse=True)
                card_detections = card_detections[:3]
                # 重新按x坐标排序
                card_detections.sort(key=lambda d: d['x'])
            
            numbers = [d['text'] for d in card_detections[:3]]
            
            if self.debug:
                print(f"  OCR识别数字: {numbers}")
            
            return numbers
            
        except Exception as e:
            if self.debug:
                print(f"  OCR失败: {e}")
            return None
    
    def extract_light_color_with_vlm(self, image_path: str) -> Optional[str]:
        """
        使用VLM识别灯光颜色
        
        Returns:
            "GREEN" | "RED" | "YELLOW" 或 None
        """
        prompt = """Look at this game screenshot carefully.

At the TOP of the screen, there are 3 circular traffic light indicators arranged horizontally.
ONLY ONE of them is lit up (bright and glowing).

Task: Identify which light is currently ON.

The lit light will be one of these colors:
- GREEN (bright green, glowing)
- RED (bright red, glowing)  
- YELLOW (bright yellow/amber, glowing)

The other two lights will be dark/dim (turned off).

Output ONLY the color name of the lit light: GREEN, RED, or YELLOW

Your answer:"""
        
        response = self.call_vlm(image_path, prompt)
        
        if self.debug:
            print(f"  Light color VLM response: {response}")
        
        # 提取颜色
        response_upper = response.upper()
        if "GREEN" in response_upper:
            return "GREEN"
        elif "RED" in response_upper:
            return "RED"
        elif "YELLOW" in response_upper:
            return "YELLOW"
        
        return None
    
    def extract_question_info(self, question_image_path: str) -> Optional[Dict]:
        """
        从question截图中提取信息
        
        Returns:
            {
                "light_color": "GREEN" | "RED" | "YELLOW",
                "numbers": [5, 8, 3]  # left, middle, right
            }
        """
        # 1. 提取数字
        if self.use_ocr:
            numbers = self.extract_numbers_with_ocr(question_image_path)
        else:
            # 使用VLM提取（备用方案）
            numbers = self.extract_numbers_with_vlm(question_image_path)
        
        if numbers is None or len(numbers) != 3:
            print(f"⚠ 无法提取3个数字: {question_image_path}")
            return None
        
        # 2. 提取灯光颜色
        light_color = self.extract_light_color_with_vlm(question_image_path)
        
        if light_color is None:
            print(f"⚠ 无法识别灯光颜色: {question_image_path}")
            return None
        
        return {
            "light_color": light_color,
            "numbers": numbers
        }
    
    def extract_numbers_with_vlm(self, image_path: str) -> Optional[List[int]]:
        """
        使用VLM提取数字（备用方案）
        
        Returns:
            [left, middle, right] 或 None
        """
        prompt = """Look at the three number cards in this game screenshot.
From LEFT to RIGHT, what are the three numbers?
Output ONLY the numbers in format: [number1, number2, number3]"""
        
        response = self.call_vlm(image_path, prompt)
        
        if self.debug:
            print(f"  Numbers VLM response: {response}")
        
        # 提取数字
        match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+)\]', response)
        if match:
            return [int(match.group(i)) for i in range(1, 4)]
        
        return None
    
    def extract_correct_answer(self, result_image_path: str) -> Optional[int]:
        """
        从result截图中提取正确答案的index
        
        Returns:
            0, 1, 或 2 (对应left, middle, right)
        """
        prompt = """Look at this game result screenshot carefully.

IMPORTANT VISUAL CUES:
- CORRECT answer: The number card highlighted with GREEN color/border
- WRONG answer: The number card highlighted with RED color/border  
- Special case: There may be TWO green cards if numbers are duplicated (you can choose either one)

Your task:
1. Find which card(s) have GREEN highlighting (this is the correct answer)
2. Determine its position from left to right
3. Output the position number:
   - Left card = 0
   - Middle card = 1
   - Right card = 2

Rules:
- If multiple cards are green (duplicated numbers), choose ANY ONE of them
- Output ONLY a single digit: 0, 1, or 2
- Do not include any explanation or other text

Your answer:"""
        
        response = self.call_vlm(result_image_path, prompt)
        
        if self.debug:
            print(f"Result VLM response: {response}")
        
        # 提取数字
        match = re.search(r'[012]', response)
        if match:
            return int(match.group(0)), response  # 返回答案和原始响应
        
        print(f"⚠ 无法从result截图提取正确答案: {result_image_path}")
        print(f"  VLM响应: {response}")
        return None, response
    
    def generate_problem_text(self, light_color: str, numbers: List[int]) -> str:
        """
        生成problem文本
        
        Args:
            light_color: 灯光颜色 (GREEN/RED/YELLOW)
            numbers: 3个数字 [left, middle, right]
            
        Returns:
            problem文本，格式: "<image>Light: GREEN. Numbers: 5, 8, 3. Select the correct one."
        """
        return f"<image>Light: {light_color}. Numbers: {numbers[0]}, {numbers[1]}, {numbers[2]}. Select the correct one."
    
    def annotate_round(
        self,
        question_image_path: str,
        result_image_path: str
    ) -> Optional[tuple]:
        """
        标注一轮游戏数据
        
        Returns:
            (sample_dict, vlm_response_dict) 或 None
        """
        # 1. 从question提取信息
        question_info = self.extract_question_info(question_image_path)
        if question_info is None:
            return None
        
        # 2. 从result提取正确答案（同时获取VLM原始响应）
        result = self.extract_correct_answer(result_image_path)
        if result is None or result[0] is None:
            return None
        
        correct_index, vlm_response = result
        
        # 3. 生成problem文本
        problem_text = self.generate_problem_text(
            question_info["light_color"],
            question_info["numbers"]
        )
        
        # 4. 构建训练样本
        sample = {
            "images": [question_image_path],
            "problem": problem_text,
            "answer": str(correct_index)
        }
        
        # 5. 记录VLM原始响应用于质量过滤
        debug_info = {
            "result_vlm_response": vlm_response
        }
        
        return (sample, debug_info)
    
    def is_high_confidence_sample(self, sample: Dict, debug_info: Dict) -> bool:
        """
        判断样本是否高置信度（VLM响应质量好）
        
        Args:
            sample: 标注样本
            debug_info: 包含VLM原始响应的调试信息
            
        Returns:
            True if high confidence
        """
        # 检查answer是否干净（只包含0/1/2，无额外文字）
        answer = sample.get("answer", "")
        if answer not in ["0", "1", "2"]:
            return False
        
        # 如果有原始VLM响应，检查其质量
        if "result_vlm_response" in debug_info:
            response = debug_info["result_vlm_response"].strip()
            
            # 最佳情况：响应就是纯数字
            if response in ["0", "1", "2"]:
                return True
            
            # 可接受：响应很短且包含数字
            if len(response) <= 10 and any(c in response for c in ["0", "1", "2"]):
                return True
            
            # 不可接受：响应太长或包含解释文字
            if len(response) > 20:
                return False
        
        return True
    
    def load_collected_data(self, data_dir: str) -> Dict[str, List[Dict]]:
        """
        加载收集的数据
        
        Returns:
            {
                "device_1": [episode1_metadata, episode2_metadata, ...],
                "device_2": [...],
                ...
            }
        """
        data_dir_path = Path(data_dir)
        device_data = defaultdict(list)
        
        # 遍历所有设备目录
        for device_dir in data_dir_path.iterdir():
            if not device_dir.is_dir():
                continue
            
            device_id = device_dir.name
            
            # 读取该设备的所有episode metadata
            for metadata_file in sorted(device_dir.glob("episode_*_metadata.json")):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    episode_data = json.load(f)
                    device_data[device_id].append(episode_data)
        
        return dict(device_data)
    
    def annotate_all_data(
        self,
        data_dir: str,
        output_dir: str,
        train_ratio: float = 0.8
    ) -> Tuple[int, int]:
        """
        标注所有收集的数据并生成训练集和测试集
        
        Args:
            data_dir: 收集数据的根目录
            output_dir: 输出目录
            train_ratio: 训练集比例（默认0.8）
            
        Returns:
            (训练样本数, 测试样本数)
        """
        print("="*60)
        print("开始标注游戏数据")
        print("="*60)
        
        # 1. 加载收集的数据
        print(f"\n加载数据从: {data_dir}")
        device_data = self.load_collected_data(data_dir)
        
        total_episodes = sum(len(episodes) for episodes in device_data.values())
        print(f"找到 {len(device_data)} 个设备，共 {total_episodes} 局游戏")
        
        # 2. 标注所有轮次
        all_samples = []
        low_confidence_samples = []
        failed_rounds = 0
        
        for device_id, episodes in device_data.items():
            print(f"\n处理设备: {device_id}")
            
            for episode in tqdm(episodes, desc=f"标注 {device_id}"):
                episode_id = episode["episode_id"]
                
                for round_data in episode["rounds"]:
                    round_num = round_data["round"]
                    question_img = round_data["question_screenshot"]
                    result_img = round_data["result_screenshot"]
                    
                    # 转换为绝对路径
                    question_img_abs = Path(data_dir) / question_img
                    result_img_abs = Path(data_dir) / result_img
                    
                    # 检查文件存在
                    if not question_img_abs.exists() or not result_img_abs.exists():
                        print(f"⚠ 文件不存在: ep{episode_id} round{round_num}")
                        failed_rounds += 1
                        continue
                    
                    # 标注
                    result = self.annotate_round(str(question_img_abs), str(result_img_abs))
                    
                    if result is not None:
                        sample, debug_info = result
                        
                        # 质量过滤
                        if self.is_high_confidence_sample(sample, debug_info):
                            all_samples.append(sample)
                        else:
                            low_confidence_samples.append(sample)
                            if self.debug:
                                print(f"  ⚠ 低置信度样本，已过滤: {debug_info.get('result_vlm_response', '')}")
                    else:
                        failed_rounds += 1
        
        print(f"\n标注完成:")
        print(f"  高质量样本: {len(all_samples)} 轮")
        print(f"  低质量样本（已过滤）: {len(low_confidence_samples)} 轮")
        print(f"  标注失败: {failed_rounds} 轮")
        
        if len(all_samples) == 0:
            print("⚠ 没有成功标注的数据，退出")
            return 0, 0
        
        # 3. 划分训练集和测试集
        import random
        random.shuffle(all_samples)
        
        split_index = int(len(all_samples) * train_ratio)
        train_samples = all_samples[:split_index]
        test_samples = all_samples[split_index:]
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_samples)} 样本")
        print(f"  测试集: {len(test_samples)} 样本")
        
        # 4. 保存JSONL文件
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_file = output_path / "train.jsonl"
        test_file = output_path / "test.jsonl"
        
        # 写入训练集
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # 写入测试集
        with open(test_file, 'w', encoding='utf-8') as f:
            for sample in test_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n数据集已保存:")
        print(f"  训练集: {train_file}")
        print(f"  测试集: {test_file}")
        
        # 5. 保存统计信息
        stats = {
            "total_samples": len(all_samples),
            "low_confidence_filtered": len(low_confidence_samples),
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "failed_rounds": failed_rounds,
            "train_ratio": train_ratio,
            "devices": list(device_data.keys()),
            "train_file": str(train_file),
            "test_file": str(test_file)
        }
        
        stats_file = output_path / "annotation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"  统计信息: {stats_file}")
        
        print("\n" + "="*60)
        print("标注完成！")
        print("="*60)
        
        return len(train_samples), len(test_samples)


def main():
    parser = argparse.ArgumentParser(description="游戏数据标注脚本（输出训练集和测试集）")
    
    # 输入输出配置
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="收集数据的根目录（包含各设备子目录）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="game_dataset",
        help="输出目录（默认 game_dataset）"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例（默认0.8）"
    )
    
    # VLM配置
    parser.add_argument(
        "--vlm-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama VLM服务地址（默认 http://localhost:11434）"
    )
    
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="qwen2.5-vl:latest",
        help="VLM模型名称（默认 qwen2.5-vl:latest）"
    )
    
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="禁用OCR，使用VLM提取数字（不推荐）"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="开启调试模式"
    )
    
    args = parser.parse_args()
    
    # 创建标注器
    annotator = GameDataAnnotator(
        vlm_base_url=args.vlm_url,
        vlm_model=args.vlm_model,
        use_ocr=not args.no_ocr,
        debug=args.debug
    )
    
    # 执行标注
    train_count, test_count = annotator.annotate_all_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio
    )
    
    print(f"\n最终输出:")
    print(f"  训练样本: {train_count}")
    print(f"  测试样本: {test_count}")
    print(f"  输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
