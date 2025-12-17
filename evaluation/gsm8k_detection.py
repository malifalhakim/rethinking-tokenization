import os
import sys
import json
import time
import logging
from typing import Optional
from dataclasses import dataclass

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from google import genai
from google.genai import types
from tqdm import tqdm

from utils.helper import prepare_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Structured result from entity detection."""
    target_word: Optional[str]
    is_safe: bool
    reason: str
    raw_response: str


class GeminiDetector:
    """Detects safe target entities in GSM8K math problems for word replacement."""
    
    SYSTEM_INSTRUCTION = """You are an expert Computational Linguist and Data Augmentation Specialist. Your task is to analyze math word problems from the GSM8K dataset and identify a single "Safe Target Entity" that can be replaced with a nonsense word without breaking the mathematical logic or semantic structure of the problem.

    ### GOAL
    Identify one specific noun or proper noun in the text that acts as a "Surface Entity" (a person, object, or place) and is NOT a mathematical variable, unit of measurement, or logical operator.

    ### SELECTION PRIORITY (Hierarchy of Safety)
    1. **Proper Names** (Highest Priority): Names of people or specific places (e.g., "Natalia", "James", "Seattle"). These are always safe to replace.
    2. **Concrete Physical Objects**: Items being counted or manipulated (e.g., "apples", "clips", "toys", "cards").
    3. **Generic Roles**: People described by role (e.g., "friends", "students", "customers").

    ### STRICT EXCLUSION LIST (Do NOT Select)
    - **Math Terms:** total, sum, difference, product, quotient, remainder, average, mean, median, ratio.
    - **Value Terms:** cost, price, amount, value, rate, speed, capacity, weight, height, distance.
    - **Units of Measurement:** hours, minutes, seconds, days, weeks, months, years, dollars, cents, degrees, miles, km, kg, lbs.
    - **Logical Connectors:** part, piece, half, third, quarter, times.

    ### OUTPUT FORMAT
    You must return a valid JSON object with the following fields:
    - "target_word": The exact substring from the original text to be replaced. If multiple valid options exist, pick the safest one (highest priority).
    - "is_safe": Boolean (true/false).
    - "reason": A brief explanation of why this word is safe or why no safe word was found.

    If NO safe word exists (e.g., the problem is purely abstract like "What is 5 + 5?"), return "target_word": null."""

    PROMPT_TEMPLATE = """Analyze the following math problem text. Identify the safest target entity to replace according to the guidelines.

    Input Text:
    \"""
    {problem_text}
    \"""

    Return ONLY the JSON object."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        thinking_budget: int = 256,
        request_delay: float = 0.1
    ):
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise ValueError("GENAI_API_KEY environment variable is not set")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.request_delay = request_delay

    def _prepare_prompt(self, problem_text: str) -> str:
        """Prepare the prompt for the API call."""
        return self.PROMPT_TEMPLATE.format(problem_text=problem_text)

    def _parse_response(self, response_text: str) -> DetectionResult:
        """Parse the API response into a structured result."""
        try:
            # Clean response text (remove markdown code blocks if present)
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            return DetectionResult(
                target_word=data.get("target_word"),
                is_safe=data.get("is_safe", False),
                reason=data.get("reason", ""),
                raw_response=response_text
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return DetectionResult(
                target_word=None,
                is_safe=False,
                reason=f"Parse error: {e}",
                raw_response=response_text
            )

    def detect(self, problem_text: str, max_retries: int = 3) -> DetectionResult:
        """Detect safe target entity in a math problem."""
        prompt = self._prepare_prompt(problem_text)
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        system_instruction=self.SYSTEM_INSTRUCTION,
                        thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget)
                    )
                )
                return self._parse_response(response.text)
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return DetectionResult(
                        target_word=None,
                        is_safe=False,
                        reason=f"API error after {max_retries} attempts: {e}",
                        raw_response=""
                    )

    def process_dataset(
        self,
        dataset,
        output_path: str,
        save_interval: int = 50
    ) -> list[dict]:
        """Process entire dataset with progress tracking and incremental saving."""
        results = []
        
        for idx, entry in enumerate(tqdm(dataset, desc="Processing problems")):
            problem_text = entry['question']
            detection = self.detect(problem_text)
            
            result = {
                "problem_text": problem_text,
                "target_word": detection.target_word,
                "is_safe": detection.is_safe,
                "reason": detection.reason,
                "raw_response": detection.raw_response
            }
            results.append(result)
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            # Incremental save
            if (idx + 1) % save_interval == 0:
                self._save_results(results, output_path)
                logger.info(f"Checkpoint saved at {idx + 1} entries")
        
        # Final save
        self._save_results(results, output_path)
        return results

    @staticmethod
    def _save_results(results: list[dict], output_path: str) -> None:
        """Save results to JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    """Main entry point for GSM8K entity detection."""
    output_path = os.path.join(
        os.path.dirname(__file__),
        "gsm8k_gemini_detection_results.jsonl"
    )
    
    logger.info("Initializing detector...")
    detector = GeminiDetector()
    
    logger.info("Loading dataset...")
    dataset = prepare_dataset(
        dataset_name="openai/gsm8k",
        dataset_subset="main",
        dataset_split="test",
        prompt_col="question"
    )
    
    logger.info(f"Processing {len(dataset)} problems...")
    results = detector.process_dataset(dataset, output_path)
    
    # Summary statistics
    safe_count = sum(1 for r in results if r["is_safe"])
    logger.info(f"Detection complete. Safe entities found: {safe_count}/{len(results)}")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()