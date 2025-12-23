import os
import accelerate
import torch
import torch.nn as nn
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


from utils import build_rm_conversation, build_qwen_omni_inputs, count_parameters


class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

        self.to_reward = nn.Linear(base_model.config.text_config.hidden_size, 1).to(
            torch.bfloat16
        )

        nn.init.normal_(self.to_reward.weight, mean=0, std=0.02)
        nn.init.constant_(self.to_reward.bias, 0)

    def forward(self, inputs):
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )  # [B, L, D]
        hidden = self.to_reward(outputs.hidden_states[-1])  # [B, L, 1]

        # Since the left-padding is used, the last token is the predicted token.
        hidden_states = hidden[:, -1, :]  # [B, 1]
        return hidden_states


class RewardModelInferencePipeline:
    def __init__(self, qwen_omni_path, rm_ckpt_path):
        self.qwen_omni_path = qwen_omni_path
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        print(f"Loading Reward Model from {rm_ckpt_path}...")
        self.model = self.build_rm_model(rm_ckpt_path)
        print(f"#Params of Reward Model: {count_parameters(self.model)}")

        print(f"Loading Qwen2.5-Omni processor from {self.qwen_omni_path}...")
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.qwen_omni_path)

    def build_rm_model(self, rm_ckpt_path):
        base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.qwen_omni_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2",
        )
        model = PeftModel.from_pretrained(
            base_model,
            os.path.join(rm_ckpt_path, "lora"),
        )

        reward_model = RewardModel(model)
        reward_model.to_reward.load_state_dict(
            torch.load(
                os.path.join(rm_ckpt_path, "reward_head.pt"), map_location=self.device
            )
        )

        reward_model.to(self.device)
        reward_model.eval()
        return reward_model

    @torch.no_grad()
    def get_pointwise_reward(self, text, wav_path):
        conversation = build_rm_conversation(wav_path, text)
        rm_inputs = build_qwen_omni_inputs(self.processor, conversation)
        rm_inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in rm_inputs.items()
        }

        with torch.no_grad():
            reward = self.model(rm_inputs)  # [1, 1]
            reward = reward.detach().cpu().tolist()
            reward = reward[0][0]

        return float(reward)

    @torch.no_grad()
    def get_pairwise_rewards(self, text, wav_path1, wav_path2):
        conversation1 = build_rm_conversation(wav_path1, text)
        conversation2 = build_rm_conversation(wav_path2, text)

        conversations = [conversation1, conversation2]
        rm_inputs = build_qwen_omni_inputs(self.processor, conversations)
        rm_inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in rm_inputs.items()
        }

        with torch.no_grad():
            rewards = self.model(rm_inputs)  # [2, 1]
            rewards = rewards.detach().cpu().tolist()
            rewards = [r[0] for r in rewards]

        return float(rewards[0]), float(rewards[1])
