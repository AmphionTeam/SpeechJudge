import torch
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)


from utils import (
    build_cot_conversation,
    build_qwen_omni_inputs,
    download_speechjudge_grm,
    count_parameters,
    extract_rating,
)


def load_model(model_path, is_omni=True):
    if is_omni:
        qwen_cls = Qwen2_5OmniForConditionalGeneration
    else:
        qwen_cls = Qwen2_5OmniThinkerForConditionalGeneration

    print("Downloading model to {}...".format(model_path))
    download_speechjudge_grm(model_path)

    print("Loading model...")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    model = qwen_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # print(model)
    print(f"#Params of Model: {count_parameters(model)}")
    return model, processor


def compare_wavs(processor, model, target_text, wav_path_a, wav_path_b, is_omni=True):
    conversion = build_cot_conversation(target_text, wav_path_a, wav_path_b)
    omni_inputs = build_qwen_omni_inputs(processor, conversion)

    omni_inputs = omni_inputs.to(model.device).to(model.dtype)
    prompt_length = omni_inputs["input_ids"].shape[1]

    if is_omni:
        text_ids = model.generate(
            **omni_inputs,
            use_audio_in_video=False,
            do_sample=True,
            return_audio=False,
        )  # [1, T]
    else:
        text_ids = model.generate(
            **omni_inputs,
            use_audio_in_video=False,
            do_sample=True,
            max_new_tokens=1024,
            eos_token_id=[151645],
            pad_token_id=151643,
        )  # [1, T]
    text_ids = text_ids[:, prompt_length:]

    text = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    rating, result = extract_rating(text[0])
    return rating, result


if __name__ == "__main__":
    model_path = "pretrained/SpeechJudge-GRM"
    model, processor = load_model(model_path)

    target_text = "The worn leather, once supple and inviting, now hangs limp and lifeless. Its time has passed, like autumn leaves surrendering to winter's chill. I shall cast it aside, making way for new beginnings and fresh possibilities."
    wav_path_a = "examples/wav_a.wav"
    wav_path_b = "examples/wav_b.wav"

    rating, result = compare_wavs(processor, model, target_text, wav_path_a, wav_path_b)

    score_A = rating["output_a"]
    score_B = rating["output_b"]
    final_result = "A" if score_A > score_B else "B" if score_A < score_B else "Tie"

    print(f"\n[Final Result] {final_result}")
    print(f"Score of Audio A: {score_A}, Score of Audio B: {score_B}")
    print("\n", "-" * 15, f"Details", "-" * 15, "\n")
    print(result)
