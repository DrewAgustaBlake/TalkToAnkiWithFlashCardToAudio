#!/usr/bin/env python3
"""
Anki Voice Reader:
- Fetches cards from a deck via AnkiConnect
- Reads the card text
- Describes images using OpenAI's vision models
- Uses OpenAI TTS to speak everything out loud

Requirements:
    pip install openai requests sounddevice

Make sure:
- Anki is running
- AnkiConnect add-on is installed and enabled
"""

import os
import sys
import json
import base64
import time
import subprocess
import re
import html
from typing import List, Dict, Any, Optional

import requests
from openai import OpenAI
import tkinter as tk
from tkinter import simpledialog

import tempfile
import wave

import sounddevice as sd
import numpy as np  # (not strictly needed, but fine to keep)


# ---------- TEMP AUDIO PATHS (ONE PER SIDE) ----------

TEMP_Q = os.path.join(tempfile.gettempdir(), "anki_temp_question.wav")
TEMP_A = os.path.join(tempfile.gettempdir(), "anki_temp_answer.wav")


# ---------- API KEY PROMPT ----------

def prompt_for_api_key() -> str:
    """
    Show a GUI dialog asking the user for their OpenAI API key.
    The text is masked (like a password).
    Returns the key as a string, or exits if cancelled.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    api_key = simpledialog.askstring(
        "OpenAI API Key",
        "Please enter your OpenAI API key:",
        show="*"  # Mask input like a password
    )

    root.destroy()

    if not api_key:
        print("[ERROR] No API key entered. Exiting.")
        sys.exit(1)

    return api_key


# ---------- SPEECH (MIC) HELPERS ----------
def play_bling():
    """Play the soft bling sound once (works on Windows or ffplay)."""
    bling_path = "/mnt/data/f22087fa-0ab0-4ee1-88a3-984008ccd2ac.wav"

    if sys.platform.startswith("win"):
        try:
            import winsound
            winsound.PlaySound(bling_path, winsound.SND_FILENAME)
            return
        except Exception as e:
            print(f"[WARN] winsound failed for bling: {e}")

    # fallback to ffplay
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", bling_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        print("[WARN] Could not play bling sound; please install ffmpeg/ffplay.")

def mark_card_reviewed(card_id: int, ease: int = 3) -> None:
    """
    Mark a card as reviewed in Anki without crashing the program.
    ease:
      1 = Again
      2 = Hard
      3 = Good
      4 = Easy
    """
    try:
        result = anki_request(
            "answerCards",
            cards=[card_id],
            ease=ease,
        )
        print(f"[ANKI] Marked card {card_id} as reviewed (ease={ease}). Result: {result}")
    except SystemExit:
        # anki_request() attempted to exit, so override it
        print(f"[WARN] Anki refused to answer card {card_id}. It's likely not in review/learning mode.")
    except Exception as e:
        print(f"[WARN] Unexpected error marking card {card_id} reviewed: {e}")


def record_audio_to_file(path: str, duration: float = 4.0, fs: int = 16000) -> None:
    """
    Record audio from the default microphone for `duration` seconds
    and save it as a mono WAV file at `fs` Hz.
    """
    play_bling()
    print("[VOICE] Listening... (speak now)")
    samples = int(duration * fs)
    audio = sd.rec(samples, samplerate=fs, channels=1, dtype="int16")
    sd.wait()

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())


def transcribe_speech_to_text(client: OpenAI, path: str) -> str:
    """
    Send the recorded audio to OpenAI for transcription (Whisper).
    Returns lowercased text.
    """
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )
    text = getattr(resp, "text", "") or ""
    return text.strip().lower()


def classify_command_with_gpt(client: OpenAI, raw_text: str) -> str:
    """
    Use a chat model to map free-form speech text to a limited command set.
    Returns exactly one of: NEXT, STOP, REPEAT, ANSWER, OTHER.
    """
    if not raw_text:
        return "OTHER"

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a command router for a flashcard app.\n"
                        "The user says short phrases like 'next', 'go ahead', "
                        "'continue', 'repeat that', 'show me the answer', or 'stop'.\n\n"
                        "You MUST respond with exactly ONE of these tokens:\n"
                        "- NEXT   (for next card / continue / proceed / go ahead)\n"
                        "- STOP   (for stop / quit / exit / I'm done)\n"
                        "- REPEAT (for repeat / say that again / once more)\n"
                        "- ANSWER (for show me the answer / reveal answer / answer)\n"
                        "- OTHER  (for anything else)\n\n"
                        "Respond with ONLY the token, in uppercase, with no extra words."
                    ),
                },
                {
                    "role": "user",
                    "content": raw_text,
                },
            ],
            max_tokens=1,
            temperature=0.0,
        )

        cmd = resp.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"[WARN] GPT command classification failed: {e}")
        return "OTHER"

    if cmd not in {"NEXT", "STOP", "REPEAT", "ANSWER", "OTHER"}:
        return "OTHER"

    return cmd


def wait_for_voice_command(client: OpenAI) -> str:
    """
    Listen for a short voice command and interpret it.
    Returns one of: 'NEXT', 'STOP', 'REPEAT', 'ANSWER', 'OTHER'.
    """
    print("\n[VOICE] Say 'next', 'continue', 'answer', 'repeat', or 'stop'...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        record_audio_to_file(tmp_path, duration=5.0)
        transcript = transcribe_speech_to_text(client, tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    print(f"[VOICE] Heard: {transcript!r}")
    cmd = classify_command_with_gpt(client, transcript)
    print(f"[VOICE] Interpreted command: {cmd}")
    return cmd


def rewrite_for_speech(client: OpenAI, text: str, side_label: str) -> str:
    """
    Use a chat model to turn the raw flashcard text into a smoother spoken prompt.
    """
    if not text.strip():
        return text

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite flashcard content into clear, natural spoken prompts.\n"
                        "For QUESTIONS: ask them directly to the listener.\n"
                        "For ANSWERS: state them clearly and succinctly.\n"
                        "Do NOT add explanations or extra teaching; just phrase it well for audio.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": f"This is the {side_label.lower()} text to read aloud:\n{text}",
                },
            ],
            max_tokens=120,
            temperature=0.3,
        )
        spoken = resp.choices[0].message.content.strip()
        return spoken
    except Exception as e:
        print(f"[WARN] Failed to rewrite {side_label} for speech: {e}")
        return text


# ---------- CONFIG ----------

ANKI_CONNECT_URL = "http://127.0.0.1:8765"

DECK_QUERY = 'deck:"***add your deck name with the :: for subdeck and copy paste it here***" is:due'
READ_SIDE = "both"  # informational

VISION_MODEL = "gpt-4.1-mini"
TTS_MODEL = "gpt-4o-mini-tts"

IMAGE_PROMPT_STYLE = (
    "Describe this image in 1–2 concise sentences as if you're reading it "
    "to someone studying from a flashcard. Focus on what's important."
)


# ---------- ANKI HELPERS ----------

def anki_request(action: str, **params) -> Any:
    payload = {
        "action": action,
        "version": 6,
        "params": params
    }

    try:
        resp = requests.post(ANKI_CONNECT_URL, json=payload)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"[ERROR] Could not reach AnkiConnect: {e}")

    data = resp.json()

    if data.get("error"):
        raise RuntimeError(f"[ANKI ERROR] {data['error']}")

    return data.get("result")



def strip_html(html_text: str) -> str:
    # Remove <style>...</style>
    text = re.sub(
        r"<style.*?</style>",
        "",
        html_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove tags
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_image_filenames(html_text: str) -> List[str]:
    return re.findall(r'<img[^>]+src="([^"]+)"', html_text)


def retrieve_media_file(filename: str) -> Optional[bytes]:
    try:
        b64_data = anki_request("retrieveMediaFile", filename=filename)
        if b64_data is None:
            return None
        return base64.b64decode(b64_data)
    except Exception as e:
        print(f"[WARN] Could not retrieve media file '{filename}': {e}")
        return None


# ---------- AUDIO PLAYBACK ----------

def play_audio_file(path: str) -> None:
    """
    Play an audio file.

    On Windows:
        - Use the default associated app via 'start' (e.g., Groove, VLC, etc.)
    On other systems:
        - Try ffplay if available, otherwise print the path.
    """
    full_path = os.path.abspath(path)

    if sys.platform.startswith("win"):
        try:
            print(f"[INFO] Opening audio with default player: {full_path}")
            subprocess.Popen(
                ["cmd", "/c", "start", "", full_path],
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception as e:
            print(f"[WARN] Default player launch failed: {e}")

    # Non-Windows or fallback
    try:
        print(f"[INFO] Playing audio via ffplay (if installed): {full_path}")
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", full_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        print(f"[INFO] Audio saved to: {full_path}")
        print("Install ffmpeg/ffplay or open the file with your system audio player.")


# ---------- OPENAI CLIENT & TTS/VISION ----------

def get_openai_client() -> OpenAI:
    api_key = prompt_for_api_key()
    return OpenAI(api_key=api_key)


def describe_image_bytes(client: OpenAI, img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_PROMPT_STYLE},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=120,
        temperature=0.2,
    )

    description = resp.choices[0].message.content
    return description.strip()


def tts_to_file(client: OpenAI, text: str, out_path: str) -> None:
    """
    Convert text to speech using OpenAI's TTS endpoint,
    then convert output into a valid WAV file that works with Windows players.
    """
    if not text.strip():
        print("[WARN] Empty text passed to TTS; skipping audio generation.")
        return

    # 1. Get audio data from OpenAI via streaming
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice="alloy",
        input=text,
    ) as response:
        chunks = []
        for chunk in response.iter_bytes():
            chunks.append(chunk)
        audio_bytes = b"".join(chunks)

    # 2. Write temp MP3 buffer
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_mp3 = tmp.name
        tmp.write(audio_bytes)

    # 3. Convert MP3 -> WAV using ffmpeg
    converted_wav = out_path
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", tmp_mp3,
                "-ac", "1",
                "-ar", "16000",
                "-sample_fmt", "s16",
                converted_wav,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as e:
        print(f"[ERROR] ffmpeg conversion failed: {e}")
        print("[INFO] Saving raw MP3 output instead.")
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        return
    finally:
        try:
            os.remove(tmp_mp3)
        except:
            pass

    try:
        size = os.path.getsize(converted_wav)
        print(f"[DEBUG] Wrote WAV file {converted_wav} ({size} bytes)")
    except OSError:
        pass


# ---------- MAIN LOGIC ----------

def build_spoken_text_for_card_side(
    client: OpenAI,
    html_side: str,
    side_label: str = "Question",
) -> str:
    text = strip_html(html_side)
    img_filenames = extract_image_filenames(html_side)

    parts = []

    if text:
        parts.append(f"{side_label}: {text}")

    if img_filenames:
        img_descs = []
        for fname in img_filenames:
            img_bytes = retrieve_media_file(fname)
            if not img_bytes:
                continue
            try:
                desc = describe_image_bytes(client, img_bytes)
                img_descs.append(desc)
            except Exception as e:
                print(f"[WARN] Vision description failed for '{fname}': {e}")

        if img_descs:
            img_text = " ".join(img_descs)
            parts.append(f"Image description: {img_text}")

    if not parts:
        return f"{side_label}: [No readable content found]."

    return " ".join(parts)


def main():
    print("== Anki Voice Reader ==")
    print("Make sure Anki + AnkiConnect are running.")
    print(f"Using deck query: {DECK_QUERY}")
    print(f"Reading: {READ_SIDE}")
    print("Press ENTER to start. Ctrl+C to quit.")
    input()

    client = get_openai_client()

    card_ids: List[int] = anki_request("findCards", query=DECK_QUERY)
    if not card_ids:
        print("[INFO] No cards found for that query.")
        return

    print(f"[INFO] Found {len(card_ids)} cards. Starting from the first one...\n")

    card_infos: List[Dict[str, Any]] = anki_request("cardsInfo", cards=card_ids)

    for idx, info in enumerate(card_infos, start=1):
        print(f"--- Card {idx}/{len(card_infos)} ---")
        question_html = info.get("question", "")
        answer_html = info.get("answer", "")

        q_text = build_spoken_text_for_card_side(client, question_html, "Question")
        a_text = build_spoken_text_for_card_side(client, answer_html, "Answer")

        print(f"[DEBUG] Question to speak (before rewrite):\n{q_text}\n")
        print(f"[DEBUG] Answer to speak (before rewrite):\n{a_text}\n")

        q_spoken = rewrite_for_speech(client, q_text, "Question")
        a_spoken = rewrite_for_speech(client, a_text, "Answer")

        print(f"[DEBUG] Question to speak (after rewrite):\n{q_spoken}\n")
        print(f"[DEBUG] Answer to speak (after rewrite):\n{a_spoken}\n")

        # ----- QUESTION -----
        try:
            print("[INFO] Generating audio for QUESTION...")
            tts_to_file(client, q_spoken, TEMP_Q)
            print("[INFO] Playing QUESTION...")
            play_audio_file(TEMP_Q)
        except Exception as e:
            print(f"[ERROR] TTS or playback failed for QUESTION: {e}")
            print(f"[INFO] Question text was:\n{q_spoken}")

        # ---------- Wait for voice command before reading ANSWER ----------
        print("[INFO] Say 'continue' or 'answer' when you're ready for the answer.\n")

        while True:
            cmd = wait_for_voice_command(client)

            if cmd in ("NEXT", "ANSWER"):
                print("[INFO] Voice: ANSWER/CONTINUE → reading the answer now.\n")
                break  # proceed to answer

            elif cmd == "REPEAT":
                print("[INFO] Voice: REPEAT → replaying question...")
                play_audio_file(TEMP_Q)

            else:
                print("[VOICE] Say 'continue', 'answer', 'repeat', or 'stop'.")


        # ----- ANSWER -----
        try:
            print("[INFO] Generating audio for ANSWER...")
            tts_to_file(client, a_spoken, TEMP_A)
            print("[INFO] Playing ANSWER...")
            play_audio_file(TEMP_A)
        except Exception as e:
            print(f"[ERROR] TTS or playback failed for ANSWER: {e}")
            print(f"[INFO] Answer text was:\n{a_spoken}")

        # ----- VOICE CONTROLLED NEXT STEP -----
        if idx < len(card_infos):
            print("\n[INFO] Awaiting voice command for next step...\n")
            while True:
                cmd = wait_for_voice_command(client)
                if cmd == "NEXT":
                    current_card_id = card_ids[idx - 1]
                    mark_card_reviewed(current_card_id, ease=3)
                    print("[INFO] Voice: NEXT → moving to next card.\n")
                    break
                elif cmd == "CONTINUE":
                    print("[INFO] Voice: NEXT → moving to next card.\n")
                    break
                elif cmd in ("REPEAT", "ANSWER"):
                    print("[INFO] Voice command → replaying **answer only**...")
                    play_audio_file(TEMP_A)
                else:
                    print("[VOICE] Say 'next' to continue, 'repeat' to hear the answer again, or 'stop' to quit.")

    print("[INFO] Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Exiting.")
