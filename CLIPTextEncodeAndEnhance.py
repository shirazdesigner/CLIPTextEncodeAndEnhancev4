import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.model_management as model_management
from comfy.model_patcher import ModelPatcher
expansion_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "expansion"))

class Expansion:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(expansion_path)
        self.model = AutoModelForCausalLM.from_pretrained(expansion_path).eval()
        load_device = model_management.text_encoder_device() if "mps" not in torch.device("cpu").type else torch.device("cpu")
        if "cpu" not in load_device.type and model_management.should_use_fp16():
            self.model.half()
        self.patcher = ModelPatcher(self.model, load_device=load_device, offload_device=model_management.text_encoder_offload_device())
    def __call__(self, prompt, seed, max_new_tokens, do_sample, num_beams, temperature, top_k, repetition_penalty):
        model_management.load_model_gpu(self.patcher)
        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data["input_ids"] = tokenized_kwargs.data["input_ids"].to(self.patcher.load_device)
        tokenized_kwargs.data["attention_mask"] = tokenized_kwargs.data["attention_mask"].to(self.patcher.load_device)
        response = self.tokenizer.batch_decode(self.model.generate(**tokenized_kwargs, max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=num_beams, temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty), skip_special_tokens=True)
        return "".join(filter(lambda x: x not in "[]【】()（）|:：", str(response[0][len(prompt):]))).strip()

class CLIPTextEncodeAndEnhance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
                "mean": (["enable", "disable"], {"default": "enable"}),
                "expansion": (["enable", "disable"], {"default": "enable"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max_new_tokens": ("INT", {"default": 64, "min": 0, "max": 1024}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "num_beams": ("INT", {"default": 4, "min": 1, "max": 10}),
                "temperature": ("FLOAT", {"default": 4, "min": 0, "max": 10, "step": 0.01}),
                "top_k": ("INT", {"default": 8, "min": 1, "max": 100}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0, "max": 10, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", )
    FUNCTION = "encode_and_enhance"
    CATEGORY = "conditioning"

    def encode_and_enhance(self, text, clip, mean, expansion, seed, max_new_tokens, do_sample, num_beams, temperature, top_k, repetition_penalty):
        torch.manual_seed(seed)
        parts = text.split(">")
        encoded_parts = []
        if expansion == "enable":
            expansion = Expansion()
            stop_words = {"when", "which", "are", "been", "own", "an", "had", "some", "so", "not", "must", "only", "is", "these", "would", "much", "him", "few", "nor", "my", "very", "her", "have", "does", "being", "can", "should", "no", "but", "why", "this", "its", "our", "me", "little", "was", "shall", "what", "who", "any", "so", "too", "did", "may", "most", "same", "or", "a", "am", "their", "how", "your", "the", "will", "that", "if", "where", "many", "his", "do", "has", "enough", "not", "it", "was", "were", "us", "them"}
            prompt = " ".join([word for word in text.split() if word.lower() not in stop_words]).strip()
            expansion_text = expansion(prompt, seed, max_new_tokens, do_sample, num_beams, temperature, top_k, repetition_penalty).rsplit(",", 1)[0]
            print(expansion_text)
            parts[-1] += ", " + expansion_text
        for i in range(len(parts)-1):
            cond, pooled = clip.encode_from_tokens(clip.tokenize(parts[i]), return_pooled=True)
            encoded_part = [[torch.cat((cond, torch.full(cond.size(), torch.mean(cond))), 1), {"pooled_output": pooled}]] if mean == "enable" else [[cond, {"pooled_output": pooled}]]
            next_cond, next_pooled = clip.encode_from_tokens(clip.tokenize(parts[i+1]), return_pooled=True)
            next_encoded_part = [[torch.cat((next_cond, torch.full(next_cond.size(), torch.mean(next_cond))), 1), {"pooled_output": next_pooled}]] if mean == "enable" else [[next_cond, {"pooled_output": next_pooled}]]
            encoded_parts.append([torch.cat((encoded_part[0][0], next_encoded_part[0][0]), 1), encoded_part[0][1]])
        if len(parts) == 1:
            cond, pooled = clip.encode_from_tokens(clip.tokenize(parts[0]), return_pooled=True)
            encoded_parts = [[torch.cat((cond, torch.full(cond.size(), torch.mean(cond))), 1), {"pooled_output": pooled}]] if mean == "enable" else [[cond, {"pooled_output": pooled}]]
        neg_encoded_parts = [[torch.div((encoded_parts[0][0] - torch.full(encoded_parts[0][0].size(), torch.mean(encoded_parts[0][0]))), 2), encoded_parts[0][1]]]
        return (encoded_parts, neg_encoded_parts, )

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeAndEnhance": CLIPTextEncodeAndEnhance,
}