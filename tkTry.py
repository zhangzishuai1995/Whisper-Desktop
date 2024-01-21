import tkinter as tk
import argparse
import io
import speech_recognition as sr
import whisper
import torch
import io
import sys
import threading
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
import json
from tkinter import ttk


with open('dict.json', 'r', encoding='utf-8') as file:
    word_dict = json.load(file)

def update_transcription(queue):
    if not queue.empty():
        new_text = queue.get()
        transcription_text.delete('1.0', tk.END)
        transcription_text.insert(tk.END, new_text)
    root.after(250, lambda: update_transcription(queue))

print(torch.__version__)
print(torch.cuda.is_available())
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
def main(queue):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=2,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)
    def record_callback(_, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True

                phrase_time = now

                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                result = audio_model.transcribe(temp_file,fp16=torch.cuda.is_available())
                text = result['text'].strip()
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                words_in_sentence = transcription[-1].split()
                new_sentence = []
                for word in words_in_sentence:
                    punctuation = ''
                    if word[-1] in '.,!?:;':
                        punctuation = word[-1]
                        word = word[:-1]

                    if word in word_dict:
                        new_word = f"{word}({word_dict[word]}){punctuation}"
                    else:
                        new_word = word + punctuation
                    new_sentence.append(new_word)
                new_sentence_str = ' '.join(new_sentence)
                print(new_sentence_str)
                queue.put(new_sentence_str)
                print('', end='', flush=True)

                sleep(0.25)
        except KeyboardInterrupt:
            break


def start_recording():
    status_label.config(text="Listening...")
    print("Start")
    root.after(250, lambda: update_transcription(transcription_queue))
    threading.Thread(target=main, args=(transcription_queue,)).start()

def stop_recording():
    # 这里添加停止录音的逻辑
    status_label.config(text="Stopped")
    print("Recording stopped")

if __name__ == "__main__":
    transcription_queue = Queue()
    root = tk.Tk()
    root.title("HeldHelp")
    root.geometry("400x300")

    status_label = tk.Label(root, text="Not Recording")  # 新增的状态标签
    status_label.pack(pady=10)

    record_button = ttk.Button(root, text="Start", command=start_recording)
    record_button.pack(pady=10)

    transcription_text = tk.Text(root, height=10, width=40)
    transcription_text.pack()

    root.mainloop()