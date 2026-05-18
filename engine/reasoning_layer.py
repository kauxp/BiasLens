import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq

class BiasReasoningEngine:
    """
    Multimodal Bias Reasoning Engine powered by Gemma.
    Analyzes fused context (Text + Image + URL + Retrieved Evidence) to detect bias.
    """
    
    def __init__(self, text_model_id="google/gemma-2b-it", vision_model_id="google/paligemma-3b-mix-224"):
        # Replace with 'google/gemma-4-it' and 'google/gemma-4-multimodal-it' when available.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # For a free HF Space, we load in lower precision if on GPU
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            print(f"Loading Text Model {text_model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_id)
            self.text_model = AutoModelForCausalLM.from_pretrained(
                text_model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            ).to(self.device)

            print(f"Loading Vision Model {vision_model_id} on {self.device}...")
            self.processor = AutoProcessor.from_pretrained(vision_model_id)
            self.vision_model = AutoModelForVision2Seq.from_pretrained(
                vision_model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            
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
                    generated_ids = self.vision_model.generate(**inputs, max_new_tokens=512)
                generated_ids = generated_ids[:, input_len:]
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt"
                ).to(self.device)
                
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    generated_ids = self.text_model.generate(**inputs, max_new_tokens=512)
                generated_ids = generated_ids[:, input_len:]
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract JSON block
            return self._parse_json(generated_text)
            
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}"}

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
            
            return {"error": "Could not parse JSON", "raw": text}
        except Exception as e:
            return {"error": f"JSON parsing error: {str(e)}", "raw": text}

    def _mock_response(self, context: dict) -> dict:
        """Returns a polished mock response for UI testing."""
        return {
            "input_type": "mixed" if context.get("has_image") and context.get("has_text") else ("image" if context.get("has_image") else "text"),
            "target_group": "Unspecified Group",
            "is_bias": True,
            "bias_types": ["Stereotyping", "Emotional Manipulation"],
            "bias_score": 0.75,
            "retrieved_evidence": [
                "Political framing often uses loaded words.",
                "Generalizations observed in the input text."
            ],
            "reasoning": "The input utilizes loaded language intended to elicit an emotional response rather than presenting objective facts. It generalizes the behavior of an entire group based on the actions of a few, which is a hallmark of stereotyping.",
            "confidence": 0.88,
            "mode": "offline (mock)"
        }
