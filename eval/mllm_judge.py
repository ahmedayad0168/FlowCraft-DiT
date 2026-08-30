import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = (
    "You are an expert AI image evaluation judge. Evaluate the image against the "
    "prompt and output strict JSON with:\n"
    "{\n"
    '  "prompt_alignment": score from 1 to 10,\n'
    '  "aesthetic_quality": score from 1 to 10,\n'
    '  "artifacts_score": score from 1 to 10,\n'
    '  "reasoning": "short explanation"\n'
    "}"
)


class MLLMJudge:
    """Vision-LLM-as-a-judge scoring of generated images."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI  # imported lazily so the package stays optional

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key: pass api_key= or set the OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    @staticmethod
    def _data_url(image_path: str) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def evaluate(self, image_path: str, prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Target Prompt: '{prompt}'"},
                        {"type": "image_url", "image_url": {"url": self._data_url(image_path)}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        content = response.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            return {"error": f"judge returned non-JSON content: {exc}", "raw": content}

    def evaluate_many(self, samples: List[Dict[str, str]]) -> Dict[str, Any]:
        """samples: [{"image_path": ..., "prompt": ...}, ...]."""
        results = [
            {**sample, **self.evaluate(sample["image_path"], sample["prompt"])}
            for sample in samples
        ]
        scored = [r for r in results if "prompt_alignment" in r]
        aggregate = {}
        if scored:
            for key in ("prompt_alignment", "aesthetic_quality", "artifacts_score"):
                values = [float(r[key]) for r in scored if key in r]
                if values:
                    aggregate[f"mean_{key}"] = sum(values) / len(values)
        return {"results": results, "aggregate": aggregate}
