def download_hugginface_model(repo_id, local_dir):
    from huggingface_hub import snapshot_download

    print("Downloading model {} to {}...".format(repo_id, local_dir))
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        resume_download=True,
        local_dir_use_symlinks=False,
    )


def build_qwen_omni_inputs(processor, conversations):
    """
    conversations:
        a list that contains B elements
    inputs:
        input_ids: torch.Size([B, T])
        attention_mask: torch.Size([B, T])
        feature_attention_mask: torch.Size([B * 1, 30000]), assuming that the audio paths of each conversion is only one
        input_features: torch.Size([B * 1, 128, 30000]), assuming that the audio paths of each conversion is only one
    """
    from qwen_omni_utils import process_mm_info

    USE_AUDIO_IN_VIDEO = False

    text = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    return inputs


def build_rm_conversation(wav_path, target_text):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "We are evaluating the naturalness of Text-to-Speech model's output. The model need to generate a natural speech for the target text.",
                },
                {"type": "text", "text": f"Target text: {target_text}"},
                {"type": "text", "text": "Output:"},
                {"type": "audio", "audio": wav_path},
                {
                    "type": "text",
                    "text": "Analysis the output above, and score it with number from 1 to 10. Note that: you only need to reply me a score (such as 7) and nothing else.",
                },
            ],
        },
    ]


def build_cot_conversation(target_text, wav_path_a, wav_path_b):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "We are comparing the naturalness of two Text-to-Speech models' outputs. The models need to generate the target text.",
                },
                {"type": "text", "text": f"Target text: {target_text}"},
                {"type": "text", "text": "Output A:"},
                {"type": "audio", "audio": wav_path_a},
                {"type": "text", "text": "Output B:"},
                {"type": "audio", "audio": wav_path_b},
                {
                    "type": "text",
                    "text": "Analysis the two output above, and score them with number from 1 to 10.",
                },
                {
                    "type": "text",
                    "text": "Note: (1) Please evaluate the naturalness of both audio outputs based on the following criteria: Prosody and Intonation, Pacing and Rhythm, Articulation and Clarity, and Overall Naturalness. (2) After conducting a detailed analysis of each criterion, using the following output template to highlight your conclusion: Output A: X, Output B: X.",
                },
            ],
        },
    ]


def build_sft_conversation(target_text, wav_path_a, wav_path_b, completion):
    return {
        "prompt": build_cot_conversation(target_text, wav_path_a, wav_path_b),
        "completion": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": completion,
                    },
                ],
            },
        ],
    }


def build_swift_grpo_conversation(
    target_text, wav_path_a, wav_path_b, human_naturalness_label
):
    raw_conversation = build_cot_conversation(target_text, wav_path_a, wav_path_b)
    assert len(raw_conversation) == 2, "Conversion should have 2 elements"

    system_content = raw_conversation[0]["content"][0]["text"]
    user_content = ""
    audio_paths = []
    for item in raw_conversation[1]["content"]:
        if item["type"] == "text":
            user_content += item["text"]
        elif item["type"] == "audio":
            user_content += "<audio>"
            audio_paths.append(item["audio"])

    conversation = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "audios": audio_paths,
        "human_naturalness_label": human_naturalness_label,
    }

    return conversation


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params < 1e6:
        return f"{total_params} params"  # Parameters
    elif total_params < 1e9:
        return f"{total_params / 1e6:.5f} M"  # Millions
    else:
        return f"{total_params / 1e9:.5f} B"  # Billions


def extract_rating(result):
    import re

    regex = r"Output A: (\d+(?:\.\d+)?).*?Output B: (\d+(?:\.\d+)?)"
    matches = re.findall(regex, result.replace("**", ""), re.DOTALL)
    if matches:
        rating = {"output_a": matches[-1][0], "output_b": matches[-1][1]}
        return rating, result

    return None, result
