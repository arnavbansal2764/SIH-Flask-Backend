import os
import wave
import pyaudio
import asyncio
from flask import Flask, render_template, request
from pydub import AudioSegment
from hume import HumeStreamClient
from hume.models.config import ProsodyConfig
import speech_recognition as sr
import ollama

app = Flask(__name__)

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 30
WAVE_OUTPUT_FILENAME = os.path.join(os.getcwd(), "output.wav")  # Save in the current directory

# Initialize lists for storing results
new_list = []
emotions = []
text_segments = []

# Function to record audio
def record_audio():
    # Create PyAudio object
    p = pyaudio.PyAudio()

    # Open a new stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    # Record audio
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    # Stop the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Write the audio data to a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Speech-to-text function for full audio
def stt_full():
    recognizer = sr.Recognizer()
    with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        text_segments.append(f"Complete answer: {text}")
    except sr.UnknownValueError:
        print("Google Web Speech could not understand the audio in full answer")
    except sr.RequestError:
        print("Could not request results from Google Web Speech API for the full answer")

# Process individual segments using Hume and Google Speech-to-Text
async def process_segment(segment, segment_index):
    segment_filename = f"output_segment_{segment_index}.wav"
    segment.export(segment_filename, format="wav")

    client = HumeStreamClient("CJffluuY10Z47dNMZSMs4WQ7eBparPq0XYWJduyczGMk9OQO")
    config = ProsodyConfig()

    async with client.connect([config]) as socket:
        result = await socket.send_file(segment_filename)
        result = result['prosody']['predictions']
        result = result[0]['emotions']

    top_3_emotions = sorted(result, key=lambda x: x['score'], reverse=True)[:3]
    new_list.append(top_3_emotions)

    current_emotions = [f"{emotion['name']} : {emotion['score']}" for emotion in top_3_emotions]
    emotions.append(current_emotions)

    recognizer = sr.Recognizer()
    with sr.AudioFile(segment_filename) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        text_segments.append(f"Text for segment {segment_index}: {text}")
    except sr.UnknownValueError:
        print(f"Google Web Speech could not understand the audio in segment {segment_index}")
    except sr.RequestError:
        print(f"Could not request results from Google Web Speech API for segment {segment_index}")

# Function to generate summary based on emotions and text
def generate_summary(emotions, text_segments, question):
    prompt = f"""
You have to judge the user's answer according to what they have spoken (text) and how they have spoken (emotions). The user does not know that the text has been divided into segments so just give a summary, give tips to the user about where and how they can improve.

question : {question}

{text_segments[-1]}

{text_segments[0]}
{text_segments[1]}
{text_segments[2]}
{text_segments[3]}
{text_segments[4]}
{text_segments[5]}

Top 3 emotions for segment 0:
{emotions[0][0]}
{emotions[0][1]}
{emotions[0][2]}

Top 3 emotions for segment 1:
{emotions[1][0]}
{emotions[1][1]}
{emotions[1][2]}

Top 3 emotions for segment 2:
{emotions[2][0]}
{emotions[2][1]}
{emotions[2][2]}

Top 3 emotions for segment 3:
{emotions[3][0]}
{emotions[3][1]}
{emotions[3][2]}

Top 3 emotions for segment 4:
{emotions[4][0]}
{emotions[4][1]}
{emotions[4][2]}

Top 3 emotions for segment 5:
{emotions[5][0]}
{emotions[5][1]}
{emotions[5][2]}
"""
    output = ollama.generate(model="llama3.1", prompt=prompt)
    return output["response"]

# Main function to process audio
async def measurer():
    audio = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
    segment_length = len(audio) // 6
    audio_segments = [audio[i * segment_length:(i + 1) * segment_length] for i in range(6)]

    tasks = [process_segment(segment, i) for i, segment in enumerate(audio_segments)]
    await asyncio.gather(*tasks)

    stt_full()
    return generate_summary(emotions, text_segments, "What is a linked list?")

# Flask routes
@app.route('/interview')
def index():
    return render_template('interview.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    record_audio()  # Record audio from microphone
    result = asyncio.run(measurer())  # Run async tasks to process the audio
    return result  # Return the result to the webpage

if __name__ == '__main__':
    app.run(debug=True)