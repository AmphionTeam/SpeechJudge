from utils import download_hugginface_model
from btrm_pipeline import RewardModelInferencePipeline


if __name__ == "__main__":
    # Load the model
    qwen_omni_path = "pretrained/Qwen2.5-Omni-7B"
    model_path = "pretrained/SpeechJudge-BTRM"

    download_hugginface_model("Qwen/Qwen2.5-Omni-7B", qwen_omni_path)
    download_hugginface_model("RMSnow/SpeechJudge-BTRM", model_path)
    inference_pipeline = RewardModelInferencePipeline(qwen_omni_path, model_path)

    # The compared two speeches (and the corresponding text)
    target_text = "The worn leather, once supple and inviting, now hangs limp and lifeless. Its time has passed, like autumn leaves surrendering to winter's chill. I shall cast it aside, making way for new beginnings and fresh possibilities."
    wav_path_a = "examples/wav_a.wav"
    wav_path_b = "examples/wav_b.wav"

    # Compare the two audio outputs
    score_A, score_B = inference_pipeline.get_pairwise_rewards(
        target_text, wav_path_a, wav_path_b
    )
    final_result = "A" if score_A > score_B else "B" if score_A < score_B else "Tie"

    print(f"\n[Final Result] {final_result}")
    print(f"Score of Audio A: {score_A}, Score of Audio B: {score_B}")
