from tqdm import tqdm
from vllm import LLM, SamplingParams
import librosa

from transformers import Qwen2_5OmniProcessor

from utils import (
    build_cot_conversation,
    download_speechjudge_grm,
    extract_rating,
)


def load_model(model_path):
    print("Downloading model to {}...".format(model_path))
    download_speechjudge_grm(model_path)

    print("Loading model...")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        max_model_len=5632,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": 2},
        seed=0,
        gpu_memory_utilization=0.5,
    )
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=50, max_tokens=1024
    )
    return processor, llm, sampling_params


def compare_wavs(
    processor,
    model,
    vllm_sampling_params,
    target_text,
    wav_path_a,
    wav_path_b,
    num_of_generation=10,
):
    conversion = build_cot_conversation(target_text, wav_path_a, wav_path_b)

    text = processor.apply_chat_template(
        conversion, add_generation_prompt=True, tokenize=False
    )
    assert len(text) == 1
    text = text[0]

    audio_data = {
        "audio": [
            librosa.load(wav_path_a, sr=None),
            librosa.load(wav_path_b, sr=None),
        ]
    }
    vllm_query = {"prompt": text, "multi_modal_data": audio_data}

    vllm_outputs = model.generate(
        [
            vllm_query
            for _ in tqdm(range(num_of_generation), desc="Generating via vllm:")
        ],
        vllm_sampling_params,
    )
    assert len(vllm_outputs) == num_of_generation

    result_list = []
    for o in vllm_outputs:
        text = o.outputs[0].text
        rating, result = extract_rating(text)
        result_list.append((rating, result))

    if num_of_generation == 1:
        return result_list[0]

    return result_list


if __name__ == "__main__":
    model_path = "pretrained/SpeechJudge-GRM"
    processor, model, vllm_sampling_params = load_model(model_path)

    target_text = "The worn leather, once supple and inviting, now hangs limp and lifeless. Its time has passed, like autumn leaves surrendering to winter's chill. I shall cast it aside, making way for new beginnings and fresh possibilities."
    wav_path_a = "examples/wav_a.wav"
    wav_path_b = "examples/wav_b.wav"

    result_list = compare_wavs(
        processor,
        model,
        vllm_sampling_params,
        target_text,
        wav_path_a,
        wav_path_b,
        num_of_generation=10,  # Inference-time Scaling @ 10
    )

    audioA_scores = []
    audioB_scores = []
    cot_details = []
    for i, (rating, result) in enumerate(result_list):
        if rating is None:
            print("[Error] No rating found")
            print(result)
            continue

        a, b = rating["output_a"], rating["output_b"]

        audioA_scores.append(float(a))
        audioB_scores.append(float(b))
        cot_details.append(result)

    score_A = sum(audioA_scores) / len(audioA_scores)
    score_B = sum(audioB_scores) / len(audioB_scores)
    final_result = "A" if score_A > score_B else "B" if score_A < score_B else "Tie"

    print(f"[Final Result] {final_result}")
    print(f"Average Score of Audio A: {score_A}, Average Score of Audio B: {score_B}")
    for i, detail in enumerate(cot_details):
        print("\n", "-" * 15, f"Result {i+1}/{len(cot_details)}", "-" * 15, "\n")
        print(detail)
