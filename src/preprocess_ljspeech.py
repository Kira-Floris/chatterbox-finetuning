# import os
# import torch
# import torchaudio
# import pandas as pd
# from tqdm import tqdm
# from src.chatterbox_.tts import ChatterboxTTS, punc_norm
# from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
# from src.chatterbox_.models.s3tokenizer import S3_SR
# from src.utils import setup_logger
# from src.config import TrainConfig


# logger = setup_logger(__name__)

# def preprocess_dataset_ljspeech(config, tts_engine: ChatterboxTTS):
    
#     data = pd.read_csv(config.csv_path, sep="|", header=None, quoting=3)
    
#     os.makedirs(config.preprocessed_dir, exist_ok=True)


#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     tts_engine.ve.to(device)
#     tts_engine.s3gen.to(device)
    
#     logger.info(f"Processing dataset... Total: {len(data)}")

#     success_count = 0

#     SPEECH_STOP_ID = getattr(tts_engine.t3.hp, 'stop_speech_token', 6562)
#     for idx, row in tqdm(data.iterrows(), total=len(data), desc="Preprocessing"):
        
#         try:
            
#             filename = str(row[0])
#             if not filename.endswith(".wav"): 
#                 filename += ".wav"
            
#             wav_path = os.path.join(config.wav_dir, filename)
            
#             if not os.path.exists(wav_path): 
#                 continue


#             wav, sr = torchaudio.load(wav_path)
            
#             if wav.shape[0] > 1: 
#                 wav = wav.mean(dim=0, keepdim=True)
                
#             if sr != S3_SR:
#                 resampler = torchaudio.transforms.Resample(sr, S3_SR)
#                 wav = resampler(wav)
            
#             wav = wav.to(device)


#             with torch.no_grad():

#                 wav_np = wav.cpu().squeeze().numpy()
                
#                 spk_emb_np = tts_engine.ve.embeds_from_wavs([wav_np], sample_rate=S3_SR)
#                 speaker_emb = torch.from_numpy(spk_emb_np[0]).cpu()


#                 s_tokens, _ = tts_engine.s3gen.tokenizer(wav.unsqueeze(0))
#                 raw_speech_tokens = s_tokens.squeeze().cpu()
                
#                 stop_speech_tensor = torch.tensor([SPEECH_STOP_ID], dtype=raw_speech_tokens.dtype)
#                 speech_tokens = torch.cat([raw_speech_tokens, stop_speech_tensor], dim=0)


#                 prompt_samples = int(config.prompt_duration * S3_SR)
#                 if wav.shape[1] < prompt_samples:
#                     prompt_wav = torch.nn.functional.pad(wav, (0, prompt_samples - wav.shape[1]))
                    
#                 else:
#                     prompt_wav = wav[:, :prompt_samples]
                
#                 p_tokens, _ = tts_engine.s3gen.tokenizer(prompt_wav.unsqueeze(0))
#                 prompt_tokens = p_tokens.squeeze().cpu()


#             raw_text = str(row[2]) if len(row) > 2 else str(row[1])
            
#             clean_text = punc_norm(raw_text)

#             if config.is_turbo:
#                 token_output = tts_engine.tokenizer(clean_text, return_tensors="pt")
#                 raw_text_tokens = token_output.input_ids[0].cpu()
                
#                 if tts_engine.tokenizer.eos_token_id is not None:
#                     text_eos = torch.tensor([tts_engine.tokenizer.eos_token_id], dtype=raw_text_tokens.dtype)
#                     text_tokens = torch.cat([raw_text_tokens, text_eos], dim=0)
#                 else:
#                     text_tokens = raw_text_tokens
            
#             else:
#                 text_tokens = tts_engine.tokenizer.text_to_tokens(clean_text).squeeze(0).cpu()


#             save_path = os.path.join(config.preprocessed_dir, filename.replace(".wav", ".pt"))
            
#             torch.save({
#                 "speech_tokens": speech_tokens,
#                 "speaker_emb": speaker_emb,
#                 "prompt_tokens": prompt_tokens,
#                 "text_tokens": text_tokens
#             }, save_path)
            
#             success_count += 1
        
#         except Exception as e:
#             logger.error(f"Error ({filename}): {e}")
#             continue
        
#     logger.info(f"Preprocessing completed! Success: {success_count}/{len(data)}")
    
    

# if __name__ == "__main__":

#     cfg = TrainConfig()
    
#     if cfg.is_turbo:
#         EngineClass = ChatterboxTurboTTS
#     else:
#         EngineClass = ChatterboxTTS
    
#     logger.info(f"{EngineClass} engine starting...")
#     tts_engine = EngineClass.from_local(cfg.model_dir, device="cpu")
    
#     preprocess_dataset_ljspeech(cfg, tts_engine)


import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread

from src.chatterbox_.tts import ChatterboxTTS, punc_norm
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.s3tokenizer import S3_SR
from src.utils import setup_logger
from src.config import TrainConfig


logger = setup_logger(__name__)


# ── CPU worker: load + resample one WAV, return tensors ready for GPU ─────────

def load_wav(wav_path: str, prompt_duration: float) -> dict | None:
    """
    Runs in a thread pool (CPU-bound).
    Loads, resamples, converts to mono, and prepares prompt slice.
    Returns a dict of CPU tensors, or None if the file is missing/broken.
    """
    try:
        wav, sr = torchaudio.load(wav_path)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != S3_SR:
            wav = torchaudio.transforms.Resample(sr, S3_SR)(wav)

        prompt_samples = int(prompt_duration * S3_SR)
        if wav.shape[1] < prompt_samples:
            prompt_wav = torch.nn.functional.pad(wav, (0, prompt_samples - wav.shape[1]))
        else:
            prompt_wav = wav[:, :prompt_samples]

        return {"wav": wav, "prompt_wav": prompt_wav}

    except Exception as e:
        logger.warning(f"load_wav failed ({wav_path}): {e}")
        return None


# ── GPU batch: run ve + s3gen tokenizer on a batch of waveforms ───────────────

def process_batch_gpu(batch: list[dict], tts_engine, device: torch.device,
                      speech_stop_id: int) -> list[dict]:
    """
    batch: list of dicts with keys: wav, prompt_wav, raw_text, filename
    Returns list of dicts with keys: speech_tokens, speaker_emb,
                                     prompt_tokens, text_tokens, filename
    """
    results = []

    # Stack wavs — pad to same length for batched speaker embedding
    wavs        = [b["wav"].squeeze(0)        for b in batch]
    prompt_wavs = [b["prompt_wav"].squeeze(0) for b in batch]

    wav_lens        = [w.shape[0] for w in wavs]
    prompt_wav_lens = [p.shape[0] for p in prompt_wavs]

    max_wav_len        = max(wav_lens)
    max_prompt_wav_len = max(prompt_wav_lens)

    # Pad to same length
    wavs_padded = torch.stack([
        torch.nn.functional.pad(w, (0, max_wav_len - w.shape[0]))
        for w in wavs
    ]).unsqueeze(1).to(device)   # (B, 1, T)

    prompt_wavs_padded = torch.stack([
        torch.nn.functional.pad(p, (0, max_prompt_wav_len - p.shape[0]))
        for p in prompt_wavs
    ]).unsqueeze(1).to(device)   # (B, 1, T)

    with torch.no_grad():

        # ── Speaker embeddings (batched) ──────────────────────────────────────
        wavs_np = wavs_padded.cpu().squeeze(1).numpy()   # (B, T)
        spk_embs_np = tts_engine.ve.embeds_from_wavs(
            list(wavs_np), sample_rate=S3_SR
        )                                                 # list of np arrays

        # ── Speech tokens (batched) ───────────────────────────────────────────
        s_tokens_batch, _ = tts_engine.s3gen.tokenizer(wavs_padded)
        p_tokens_batch, _ = tts_engine.s3gen.tokenizer(prompt_wavs_padded)

    stop_tensor = torch.tensor([speech_stop_id], dtype=torch.long)

    for i, item in enumerate(batch):
        try:
            speaker_emb = torch.from_numpy(spk_embs_np[i]).cpu()

            raw_speech  = s_tokens_batch[i].cpu()
            # Trim padding tokens if the tokenizer pads output
            raw_speech  = raw_speech[:wav_lens[i] * S3_SR // S3_SR]  # no-op trim guard
            speech_tokens = torch.cat([raw_speech, stop_tensor], dim=0)

            prompt_tokens = p_tokens_batch[i].cpu()

            results.append({
                "speech_tokens": speech_tokens,
                "speaker_emb":   speaker_emb,
                "prompt_tokens": prompt_tokens,
                "filename":      item["filename"],
                "raw_text":      item["raw_text"],
            })
        except Exception as e:
            logger.error(f"GPU batch item failed ({item['filename']}): {e}")

    return results


# ── Text tokenization (CPU, runs after GPU batch) ─────────────────────────────

def tokenize_text(result: dict, tts_engine, is_turbo: bool) -> dict | None:
    try:
        clean_text = punc_norm(result["raw_text"])

        if is_turbo:
            token_output     = tts_engine.tokenizer(clean_text, return_tensors="pt")
            raw_text_tokens  = token_output.input_ids[0].cpu()
            if tts_engine.tokenizer.eos_token_id is not None:
                eos = torch.tensor(
                    [tts_engine.tokenizer.eos_token_id],
                    dtype=raw_text_tokens.dtype
                )
                text_tokens = torch.cat([raw_text_tokens, eos], dim=0)
            else:
                text_tokens = raw_text_tokens
        else:
            text_tokens = tts_engine.tokenizer.text_to_tokens(clean_text).squeeze(0).cpu()

        result["text_tokens"] = text_tokens
        return result
    except Exception as e:
        logger.error(f"Text tokenization failed ({result['filename']}): {e}")
        return None


# ── Main preprocessing function ───────────────────────────────────────────────

def preprocess_dataset_ljspeech(
    config,
    tts_engine: ChatterboxTTS,
    batch_size: int = 32,
    num_load_workers: int = 8,
):
    """
    Parallelized preprocessing:
      - ThreadPoolExecutor prefetches + loads WAVs on CPU in parallel
      - GPU processes a batch of waveforms at once
      - ThreadPoolExecutor runs text tokenization in parallel after each batch

    Args:
        batch_size:        Number of WAVs to send to GPU at once.
                           Increase until VRAM runs out (start with 16-32).
        num_load_workers:  Threads for parallel WAV loading.
                           Set to number of CPU cores (or slightly above).
    """
    data = pd.read_csv(config.csv_path, sep="\t", quoting=3)
    os.makedirs(config.preprocessed_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts_engine.ve.to(device)
    tts_engine.s3gen.to(device)

    logger.info(f"Processing {len(data):,} rows | "
                f"batch_size={batch_size} | "
                f"load_workers={num_load_workers} | "
                f"device={device}")

    SPEECH_STOP_ID = getattr(tts_engine.t3.hp, 'stop_speech_token', 6562)

    # ── Build list of pending items ───────────────────────────────────────────
    pending = []
    for idx, row in data.iterrows():
        filename = str(row[0])
        if not filename.endswith(".wav"):
            filename += ".wav"
        wav_path = os.path.join(config.wav_dir, filename)
        if not os.path.exists(wav_path):
            continue

        raw_text = str(row[2]) if len(row) > 2 else str(row[1])

        # Skip already preprocessed files
        save_path = os.path.join(
            config.preprocessed_dir, filename.replace(".wav", ".pt")
        )
        if os.path.exists(save_path):
            continue

        pending.append({
            "filename": filename,
            "wav_path": wav_path,
            "raw_text": raw_text,
            "save_path": save_path,
        })

    logger.info(f"{len(pending):,} files to process "
                f"({len(data) - len(pending):,} already done)")

    if not pending:
        logger.info("Nothing to do.")
        return

    success_count = 0
    pbar = tqdm(total=len(pending), desc="Preprocessing")

    # ── Process in batches ────────────────────────────────────────────────────
    # For each batch:
    #   1. Submit all WAV loads to thread pool (parallel I/O)
    #   2. Collect loaded wavs
    #   3. Run GPU batch (ve + s3gen)
    #   4. Run text tokenization in parallel threads
    #   5. Save .pt files in parallel threads

    for batch_start in range(0, len(pending), batch_size):
        batch_items = pending[batch_start: batch_start + batch_size]

        # ── Step 1: parallel WAV loading ──────────────────────────────────────
        loaded_batch = []
        with ThreadPoolExecutor(max_workers=num_load_workers) as pool:
            future_to_item = {
                pool.submit(load_wav, item["wav_path"], config.prompt_duration): item
                for item in batch_items
            }
            for future in as_completed(future_to_item):
                item   = future_to_item[future]
                result = future.result()
                if result is not None:
                    loaded_batch.append({
                        "wav":        result["wav"],
                        "prompt_wav": result["prompt_wav"],
                        "filename":   item["filename"],
                        "raw_text":   item["raw_text"],
                        "save_path":  item["save_path"],
                    })

        if not loaded_batch:
            pbar.update(len(batch_items))
            continue

        # ── Step 2: GPU batch ─────────────────────────────────────────────────
        try:
            gpu_results = process_batch_gpu(
                loaded_batch, tts_engine, device, SPEECH_STOP_ID
            )
        except Exception as e:
            logger.error(f"GPU batch failed: {e}")
            pbar.update(len(batch_items))
            continue

        # Merge save_path back into gpu_results
        save_path_map = {item["filename"]: item["save_path"] for item in loaded_batch}
        for r in gpu_results:
            r["save_path"] = save_path_map.get(r["filename"], "")

        # ── Step 3: text tokenization + save (parallel threads) ──────────────
        def tokenize_and_save(result: dict) -> bool:
            result = tokenize_text(result, tts_engine, config.is_turbo)
            if result is None:
                return False
            try:
                torch.save(
                    {
                        "speech_tokens": result["speech_tokens"],
                        "speaker_emb":   result["speaker_emb"],
                        "prompt_tokens": result["prompt_tokens"],
                        "text_tokens":   result["text_tokens"],
                    },
                    result["save_path"],
                )
                return True
            except Exception as e:
                logger.error(f"Save failed ({result['filename']}): {e}")
                return False

        with ThreadPoolExecutor(max_workers=num_load_workers) as pool:
            futures = [pool.submit(tokenize_and_save, r) for r in gpu_results]
            for future in as_completed(futures):
                if future.result():
                    success_count += 1

        pbar.update(len(batch_items))

    pbar.close()
    logger.info(f"Preprocessing complete! "
                f"Success: {success_count:,}/{len(pending):,}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = TrainConfig()

    if cfg.is_turbo:
        EngineClass = ChatterboxTurboTTS
    else:
        EngineClass = ChatterboxTTS

    logger.info(f"{EngineClass} engine starting...")
    tts_engine = EngineClass.from_local(cfg.model_dir, device="cpu")

    preprocess_dataset_ljspeech(
        config           = cfg,
        tts_engine       = tts_engine,
        batch_size       = 32,   # ← increase until GPU VRAM runs out
        num_load_workers = 8,    # ← set to your CPU core count
    )
