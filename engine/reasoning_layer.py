import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN") or os.getenv("bias_deployment"))
class BiasReasoningEngine:
    """
    Multimodal Bias Reasoning Engine powered by Gemma.
    Analyzes fused context (Text + Image + URL + Retrieved Evidence) to detect bias.
    """

    def __init__(self, text_model_id="google/gemma-2b-it", vision_model_id="google/paligemma-3b-mix-224"):
        # Replace with 'google/gemma-4-it' and 'google/gemma-4-multimodal-it' when available.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            print(f"Loading Text Model {text_model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_id)
            self.text_model = AutoModelForCausalLM.from_pretrained(
                text_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            print(f"Loading Vision Model {vision_model_id} on {self.device}...")
            self.processor = AutoProcessor.from_pretrained(vision_model_id)
            self.vision_model = AutoModelForVision2Seq.from_pretrained(
                vision_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            self.is_loaded = True
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Falling back to Demo/Mock Mode for UI testing.")
            self.is_loaded = False

    def _build_context_fusion_prompt(self, unified_context: dict, evidence: list[str]) -> str:
        """
        --- CONTEXT FUSION LAYER ---
        Constructs the prompt forcing JSON output by combining inputs and retrieved evidence.
        [LITERT PLACEHOLDER]: For edge compatibility, this prompt builder would natively
        run in C++/Java alongside the TFLite runtime to avoid large prompt payloads from network.
        """
        
        text_content = unified_context.get('text', '')
        
        evidence_str = ""
        if evidence:
            evidence_str = "Relevant bias patterns:\n" + "\n".join([f"- {e}" for e in evidence]) + "\n\n"

        prompt = f"""You are BiasLens, an expert media literacy assistant.
Detect cognitive, linguistic, and structural biases in any text or image.
Be evidence-based (cite exact phrases from the input), measured (flag genuine bias only),
always provide a neutral rewrite, and respond in valid JSON.
If you lack sufficient context to analyse bias, set is_bias to false
and explain in summary that more context is needed.

Bias dimensions: gender | racial | age | religion | political | socioeconomic | disability | intersectional | lgbtq | nationality

Scoring guide (overall bias_score 0.0-1.0):
  0.0-0.2: No meaningful bias — factual, balanced reporting
  0.2-0.4: Mild framing — word choice could be improved
  0.4-0.6: Moderate bias — loaded language, missing perspectives
  0.6-0.8: Strong bias — systematic framing, stereotyping
  0.8-1.0: Extreme bias — dehumanizing language, hate speech

{evidence_str}INPUT:
{text_content}

First, think step-by-step in 2-3 sentences about what biases (if any) are present.
Then output your final analysis as a JSON block enclosed in ```json ... ```.

JSON schema:
{{
  "input_type": "text/image/url",
  "target_group": "identified group or none",
  "is_bias": true/false,
  "bias_score": 0.0 to 1.0,
  "bias_dimensions": {{"gender_bias": 0.0, "racial_bias": 0.0}},
  "biased_phrases": ["exact phrase from input"],
  "reasoning": "why biased",
  "evidence": ["from context"],
  "confidence": 0.0 to 1.0,
  "neutral_rewrite": "suggested unbiased version",
  "mode": "offline/online"
}}
"""
        return prompt

    def _canonical_error(self, error_msg: str, mode: str = "error") -> dict:
        """Returns a fully-populated schema dict for error/degraded states."""
        return {
            "input_type": "unknown",
            "target_group": "none",
            "is_bias": False,
            "bias_score": 0.0,
            "bias_dimensions": {},
            "biased_phrases": [],
            "reasoning": "",
            "evidence": [],
            "confidence": 0.0,
            "neutral_rewrite": "",
            "mode": mode,
            "error": error_msg,
        }

    def _normalize_report(self, report: dict) -> dict:
        """Fills missing keys with canonical defaults and coerces types."""
        defaults = {
            "input_type": "unknown",
            "target_group": "none",
            "is_bias": False,
            "bias_score": 0.0,
            "bias_dimensions": {},
            "biased_phrases": [],
            "reasoning": "",
            "evidence": [],
            "confidence": 0.0,
            "neutral_rewrite": "",
            "mode": "offline",
        }
        # Migrate legacy field names that the model may still emit
        if "retrieved_evidence" in report and "evidence" not in report:
            report["evidence"] = report.pop("retrieved_evidence")
        if "bias_types" in report and "bias_dimensions" not in report:
            report["bias_dimensions"] = {
                t.lower().replace(" ", "_") + "_bias": 0.5
                for t in report.pop("bias_types")
            }
        result = {**defaults, **report}
        # Coerce critical fields to their expected types
        if not isinstance(result["bias_dimensions"], dict):
            result["bias_dimensions"] = {}
        if not isinstance(result["biased_phrases"], list):
            result["biased_phrases"] = []
        if not isinstance(result["evidence"], list):
            result["evidence"] = []
        for numeric_key in ("bias_score", "confidence"):
            if not isinstance(result[numeric_key], (int, float)):
                try:
                    result[numeric_key] = float(result[numeric_key])
                except (ValueError, TypeError):
                    result[numeric_key] = 0.0
        return result

    def analyze(self, unified_context: dict, retrieved_evidence: list[str]) -> dict:
        """
        Runs the full multimodal reasoning pipeline.

        [LITERT PLACEHOLDER]:
        In a mobile/edge deployment, this method would interface with a
        tflite runtime (e.g. interpreter = tf.lite.Interpreter(model_path)).
        Tensors would be quantized to INT8.
        """

        prompt = self._build_context_fusion_prompt(unified_context, retrieved_evidence)
        image = unified_context.get('image')

        if not self.is_loaded:
            # Mock response for hackathon UI iteration if the environment lacks memory
            return self._mock_response(unified_context)

        try:
            # Prepare inputs and Generate
            # [LITERT PLACEHOLDER]: inference happens here via interpreter.invoke()
            if image is not None:
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    generated_ids = self.vision_model.generate(**inputs, max_new_tokens=256)
                generated_ids = generated_ids[:, input_len:]
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt"
                ).to(self.device)

                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    generated_ids = self.text_model.generate(**inputs, max_new_tokens=256)
                generated_ids = generated_ids[:, input_len:]
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Extract and normalise JSON block
            return self._normalize_report(self._parse_json(generated_text))

        except Exception as e:
            return self._canonical_error(f"Inference failed: {str(e)}")

    def _parse_json(self, text: str) -> dict:
        """Extracts JSON from the model's raw text output."""
        import re
        try:
            # 1. Try to extract from markdown blocks
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 2. Try to find JSON from the start of the first '{' to the end of the string
            start = text.find('{')
            if start != -1:
                # Iterate backwards from the last '}' found
                end = text.rfind('}')
                while end > start:
                    try:
                        json_str = text[start:end+1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        end = text.rfind('}', start, end)

            return self._canonical_error("Could not parse model output as JSON")
        except Exception as e:
            return self._canonical_error(f"JSON parsing error: {str(e)}")

    def _mock_response(self, context: dict) -> dict:
        """Returns a polished mock response for UI testing."""
        input_type = (
            "mixed" if context.get("has_image") and context.get("has_text")
            else ("image" if context.get("has_image") else "text")
        )
        return {
            "input_type": input_type,
            "target_group": "Unspecified Group",
            "is_bias": True,
            "bias_score": 0.75,
            "bias_dimensions": {
                "political_bias": 0.6,
                "racial_bias": 0.5,
                "gender_bias": 0.3,
            },
            "biased_phrases": ["always barbaric", "invading forces"],
            "reasoning": (
                "The input utilizes loaded language intended to elicit an emotional response "
                "rather than presenting objective facts. It generalizes the behavior of an "
                "entire group based on the actions of a few, which is a hallmark of stereotyping."
            ),
            "evidence": [
                "Political framing often uses loaded words.",
                "Generalizations observed in the input text.",
            ],
            "confidence": 0.88,
            "neutral_rewrite": (
                "Forces from that country have been reported to engage in actions "
                "that violate international norms."
            ),
            "mode": "offline (mock)",
        }
