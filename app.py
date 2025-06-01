from flask import Flask, request, send_from_directory, render_template
import os
import logging
import uuid
from pydub import AudioSegment
from pydub.generators import Sine
import whisper  # Import Whisper
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import wave
import threading
import concurrent.futures
import numpy as np
import io
from functools import lru_cache
import re
import scipy.signal as signal
from scipy.io import wavfile

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
SetLogLevel(-1)
MODEL_PATH = r"C:\Users\rajro\Downloads\vosk-model-en-us-0.22\vosk-model-en-us-0.22"
whisper_model = whisper.load_model("small")  # Load the Whisper model
vosk_model = None
vosk_model_loaded = False

def load_vosk_model_background():
    global vosk_model, vosk_model_loaded
    try:
        vosk_model = Model(MODEL_PATH)
        vosk_model_loaded = True
        logger.info("Loaded Vosk model successfully")
    except Exception as e:
        logger.error(f"Failed to load Vosk model: {e}")
        logger.warning("Audio word timing features will be disabled")
        vosk_model_loaded = False


threading.Thread(target=load_vosk_model_background, daemon=True).start()
BAD_WORDS = {
    "hate", "abuse", "stupid", "idiot", "fool", "damn", "jerk", "moron", "loser", "dumb", "whores",
    "bastard", "asshole", "bitch", "shit", "fuck", "cunt", "prick", "dick", "pussy", "fucking", "shitiest", "nigga",
    "whore", "slut", "faggot", "nigger", "chink", "gook", "spic", "kike", "wop", "dago",
    "retard", "cripple", "freak", "tranny", "dyke", "queer", "savage", "ape", "mongrel",
    "paki", "raghead", "towelhead", "cracker", "redneck", "hillbilly", "trash", "scum",
    "pig", "dog", "coward", "sissy", "twat", "wanker", "arse", "bollocks", "bloody",
    "piss", "arsehole", "shithead", "douche", "motherfucker", "ass", "cock", "tits",
    "coon", "jap", "wetback", "beaner", "gringo", "yid", "heeb", "mick", "kraut", "limey",
    "skank", "slag", "tard", "gimpy", "midget", "halfwit", "dolt", "cretin", "numbskull",
    "arsewipe", "tosser", "git", "prat", "knob", "bellend", "shite", "bugger", "sod",
    "poof", "pansy", "nancy", "fruit", "chav", "yokel", "bogan", "pleb", "peasant",
    "mong", "spaz", "lame", "goon", "thug", "brat", "snot", "puke", "sleaze", "perv",
    "nonce", "wog", "boong", "abbo", "darky", "slope", "slant", "cameljockey", "jihadi",
    "papist", "heathen", "infidel", "zionist", "commie", "pinko", "nazi", "femi-nazi",
    "bimbo", "ditz", "cow", "hag", "crone", "harpy", "shrew", "frigid", "prude", "cuck"
}


WORD_BOUNDARY_PATTERN = re.compile(r'\b{}\b')

@app.route('/')
def upload_form():
    return render_template('index.html')


def save_file_async(file, path):
    file.save(path)
    logger.info(f"Saved uploaded file to {path}")
    return path

@lru_cache(maxsize=16)
def generate_beep(duration_ms, sample_rate=44100):
    """Cache beep generation for common durations"""
    return Sine(1000).to_audio_segment(duration=duration_ms, volume=-3)

def convert_to_wav(audio_path, output_path):
    """Optimized audio conversion"""
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path

def fast_word_check(words):
    """Fixed bad word detection to only match exact words, not substrings"""
    bad_word_indices = []
    bad_words_found = []
    
    for i, word in enumerate(words):
        
        cleaned_word = ''.join(ch for ch in word.lower() if ch.isalpha())
        

        if cleaned_word in BAD_WORDS:
            bad_word_indices.append(i)
            bad_words_found.append(word)
            
    logger.info(f"Bad word check found {len(bad_words_found)} words: {bad_words_found}")
    return bad_word_indices, bad_words_found

@app.route('/convert', methods=['POST'])
def convert_audio():
    if 'audio' not in request.files:
        return "No file uploaded", 400

    file = request.files['audio']
    if file.filename == '':
        return "No selected file", 400

    try:
        censor_type = request.form.get('censor_type', 'beep')
        
        padding_ms = min(max(int(request.form.get('padding', 100)), 25), 300)
    except ValueError:
        padding_ms = 100  

    file_extension = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    audio_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    try:
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(save_file_async, file, audio_path)
            audio_path = future.result()

        
        wav_path = audio_path
        if file_extension != '.wav':
            try:
                wav_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.wav")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(convert_to_wav, audio_path, wav_path)
                    wav_path = future.result()
                logger.info(f"Converted to WAV: {wav_path}")
            except Exception as e:
                logger.error(f"Failed to convert audio format: {e}")
                return f"Error converting audio format: {str(e)}", 500

        
        text, confidence = transcribe_audio_whisper(wav_path) 
        if not text:
            return "Could not transcribe the audio. Please try again with a clearer recording.", 400

        
        logger.info(f"Transcribed text: {text}")
        
        words = text.split()
        display_words = words.copy()

        
        bad_word_indices, bad_words_found = fast_word_check(words)

        
        for i in bad_word_indices:
            if 0 <= i < len(display_words):
                display_words[i] = "____"

        
        if bad_words_found:
            logger.info(f"Abusive words found: {bad_words_found}")

            
            timestamps = []
            
            
            if vosk_model_loaded:
                vosk_timestamps = get_word_timestamps(wav_path, bad_words_found, padding_ms)
                if vosk_timestamps:
                    timestamps = vosk_timestamps
                    logger.info(f"Vosk found {len(timestamps)} word timestamps")
            
            
            if not timestamps:
                logger.info("Vosk timestamps not available, trying enhanced phonetic alignment")
                phonetic_timestamps = enhanced_phonetic_alignment(wav_path, text, bad_words_found, bad_word_indices, padding_ms)
                if phonetic_timestamps:
                    timestamps = phonetic_timestamps
                    logger.info(f"Enhanced phonetic alignment found {len(timestamps)} word timestamps")
            
            
            if not timestamps:
                logger.info("Trying improved forced alignment")
                forced_timestamps = improved_forced_alignment(wav_path, text, bad_words_found, bad_word_indices, padding_ms)
                if forced_timestamps:
                    timestamps = forced_timestamps
                    logger.info(f"Improved forced alignment found {len(timestamps)} word timestamps")
            
            
            if not timestamps:
                logger.warning("All precise methods failed, using time-based estimation with energy analysis")
                audio_duration = AudioSegment.from_wav(wav_path).duration_seconds
                estimated_timestamps = improved_word_timestamp_estimation(wav_path, words, bad_words_found, 
                                                                         bad_word_indices, audio_duration, padding_ms)
                timestamps = estimated_timestamps
            
            
            timestamps = advanced_energy_based_refinement(wav_path, timestamps, padding_ms)
            
            
            timestamps = apply_preemptive_timing(timestamps, 25) 

            
            output_filename = f"censored_{uuid.uuid4()}.wav"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)
            success = enhanced_audio_censoring(wav_path, output_path, timestamps, censor_type)

            if not success:
                return "Failed to censor words in audio", 500
        else:
            
            output_filename = f"processed_{uuid.uuid4()}.wav"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)
            with open(wav_path, 'rb') as src_file:
                with open(output_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())
            logger.info("No abusive words found; output audio remains unchanged.")

        
        return render_template(
            'results.html',
            display_words=display_words,
            output_filename=output_filename,
            bad_words_found=bad_words_found,
            confidence=int(confidence * 100)
        )

    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        return f"Error processing audio: {str(e)}", 500

def transcribe_audio_whisper(audio_path):
    """Transcribe audio using Whisper"""
    try:
        result = whisper_model.transcribe(audio_path)
        text = result["text"]
        
        return text, 0.9  
    except Exception as e:
        logger.error(f"Error transcribing with Whisper: {e}")
        return "", 0
    
def get_word_timestamps(audio_path, target_words, padding_ms=100):
    """Enhanced timestamp detection with exact word matching and pre-emptive timing"""
    if not vosk_model_loaded:
        return []
        
    timestamps = []

    target_words_lower = [word.lower() for word in target_words]
    target_words_clean = [''.join(ch for ch in word.lower() if ch.isalpha()) for word in target_words]
    
    
    expanded_targets = set()
    for word in target_words_clean:
        expanded_targets.add(word)
        
        if len(word) > 3:
            expanded_targets.add(word[:-1])  
            expanded_targets.add(word + 's')  
            expanded_targets.add(word + 'ing') 
    
    try:
        with wave.open(audio_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != 'NONE':
                logger.warning("Audio file not in required format for Vosk")
                return []
                
            rec = KaldiRecognizer(vosk_model, wf.getframerate())
            rec.SetWords(True)
            
            
            chunk_size = 4096
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    process_enhanced_results(result, target_words_lower, target_words_clean, expanded_targets, timestamps, padding_ms)
                    
            result = json.loads(rec.FinalResult())
            process_enhanced_results(result, target_words_lower, target_words_clean, expanded_targets, timestamps, padding_ms)
            
        return timestamps
    except Exception as e:
        logger.error(f"Error in Vosk processing: {e}", exc_info=True)
        return []

def process_enhanced_results(result, target_words_lower, target_words_clean, expanded_targets, timestamps, padding_ms=100):
    """Process recognition results with enhanced word matching for better accuracy"""
    if 'result' not in result:
        return
        
    
    target_set_original = set(target_words_lower)
    target_set_clean = set(target_words_clean)
    word_confidences = {}
    
    for word_data in result['result']:
        word = word_data.get('word', '').lower()
        word_clean = ''.join(ch for ch in word if ch.isalpha())
        confidence = word_data.get('conf', 0.0)
        
        
        is_match = False
        matching_target = None
        
        
        if word in target_set_original or word_clean in target_set_clean:
            is_match = True
            # Find the original target word
            for i, target in enumerate(target_words_lower):
                if target == word or target_words_clean[i] == word_clean:
                    matching_target = target_words_lower[i]
                    break
        
        
        elif word_clean in expanded_targets:
            is_match = True
            
            for i, target in enumerate(target_words_clean):
                if target in word_clean or word_clean in target:
                    matching_target = target_words_lower[i]
                    break
            # If still no match, use the first target as fallback
            if not matching_target and target_words_lower:
                matching_target = target_words_lower[0]
        
        if is_match and matching_target:
            
            start_time = int(word_data['start'] * 1000)
            end_time = int(word_data['end'] * 1000)
            
            
            word_length_factor = min(1.0, max(0.5, len(word) / 8))
            actual_padding = int(padding_ms * word_length_factor)
            
            
            start_time = max(0, start_time - actual_padding - 25)  
            end_time = end_time + actual_padding
            
           
            key = (matching_target, start_time, end_time)
            if key not in word_confidences or confidence > word_confidences[key]:
                word_confidences[key] = confidence
    
    
    word_entries = {}
    for (word, start, end), conf in word_confidences.items():
        if word not in word_entries or conf > word_entries[word][2]:
            word_entries[word] = (start, end, conf)
    
    
    for word, (start, end, _) in word_entries.items():
        timestamps.append((start, end, word))

def enhanced_phonetic_alignment(audio_path, transcript, target_words, target_indices, padding_ms=100):
    """Enhanced phonetic alignment for more precise word boundaries"""
    try:
        
        sample_rate, audio_data = wavfile.read(audio_path)
        
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = audio_data / max_val
        
        # Calculate total duration
        total_duration_ms = int((len(audio_data) / sample_rate) * 1000)
        
        
        words = transcript.lower().split()
        
        
        syllable_durations = {
            1: 150,  # Short words (1-3 chars)
            2: 200,  # Medium words (4-6 chars)
            3: 250,  # Longer words (7+ chars)
        }
        
        
        word_durations = []
        for word in words:
            cleaned_word = ''.join(ch for ch in word if ch.isalpha())
            
            vowels = 'aeiouy'
            v_count = sum(1 for c in cleaned_word if c in vowels)
           
            syllable_count = 0
            prev_is_vowel = False
            for c in cleaned_word:
                is_vowel = c in vowels
                if is_vowel and not prev_is_vowel:
                    syllable_count += 1
                prev_is_vowel = is_vowel
            syllable_count = max(1, syllable_count)
            
            
            if len(cleaned_word) <= 3:
                base_duration = syllable_durations[1]
            elif len(cleaned_word) <= 6:
                base_duration = syllable_durations[2]
            else:
                base_duration = syllable_durations[3]
            
            
            duration = base_duration * syllable_count
            word_durations.append(duration)
        
        
        total_estimated_duration = sum(word_durations)
        scaling_factor = total_duration_ms / total_estimated_duration
        word_durations = [int(d * scaling_factor) for d in word_durations]
        
        
        word_positions = []
        current_pos = 0
        for duration in word_durations:
            word_positions.append((current_pos, current_pos + duration))
            current_pos += duration
        
        
        timestamps = []
        for i, word_idx in enumerate(target_indices):
            if 0 <= word_idx < len(word_positions):
                word = target_words[i]
                word_start, word_end = word_positions[word_idx]
                
               
                word_length = len(word)
                length_factor = min(1.2, max(0.8, word_length / 5))
                actual_padding = int(padding_ms * length_factor)
                
                
                word_start = max(0, word_start - actual_padding - 25)  # Start a bit earlier
                word_end = min(total_duration_ms, word_end + actual_padding)
                
                
                start_sample = int((word_start / 1000) * sample_rate)
                end_sample = int((word_end / 1000) * sample_rate)
                
                
                start_sample = max(0, min(start_sample, len(audio_data) - 1))
                end_sample = max(start_sample + 1, min(end_sample, len(audio_data)))
                
                
                segment = audio_data[start_sample:end_sample]
                if len(segment) > 0:
                    
                    energy = np.square(segment)
                   
                    window_size = int(0.025 * sample_rate)  # 25ms window
                    if window_size > 1 and len(energy) > window_size:
                        smoothed_energy = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
                    else:
                        smoothed_energy = energy
                    
                    
                    if len(smoothed_energy) > 0:
                        threshold = 0.1 * np.max(smoothed_energy)
                        speech_positions = np.where(smoothed_energy > threshold)[0]
                        if len(speech_positions) > 0:
                            speech_start = speech_positions[0]
                            speech_end = speech_positions[-1]
                            
                            # Convert back to milliseconds, applying pre-emptive adjustment
                            refined_start = word_start + int((speech_start / len(segment)) * (word_end - word_start)) - 30
                            refined_end = word_start + int((speech_end / len(segment)) * (word_end - word_start)) + 20
                            
                            # Ensure valid boundaries
                            refined_start = max(0, refined_start)
                            refined_end = min(total_duration_ms, refined_end)
                            
                            timestamps.append((refined_start, refined_end, word))
                            continue
                
                
                timestamps.append((word_start, word_end, word))
                
        return timestamps
    except Exception as e:
        logger.error(f"Error in enhanced phonetic alignment: {e}", exc_info=True)
        return []

def improved_forced_alignment(audio_path, transcript, target_words, target_indices, padding_ms=100):
    """Improved forced alignment with dynamic timing adjustments"""
    try:
        # Load audio for analysis
        audio = AudioSegment.from_wav(audio_path)
        sample_rate = audio.frame_rate
        total_duration_ms = len(audio)
        
        # Split transcript into words
        words = transcript.split()
        
        # Calculate initial word timing using dynamic word-length based approach
        word_durations = []
        for word in words:
            # Estimate word duration based on length and complexity
            cleaned_word = ''.join(ch for ch in word if ch.isalpha())
            # Base duration in ms (approximately 70ms per character with minimum 150ms)
            duration = max(150, len(cleaned_word) * 70)
            word_durations.append(duration)
        
        # Adjust to match total duration
        total_estimated_duration = sum(word_durations)
        if total_estimated_duration > 0:  # Prevent division by zero
            scaling_factor = total_duration_ms / total_estimated_duration
            word_durations = [int(d * scaling_factor) for d in word_durations]
        
        # Calculate cumulative positions
        positions = []
        current_pos = 0
        for duration in word_durations:
            positions.append((current_pos, current_pos + duration))
            current_pos += duration
        
        # Apply advanced energy analysis for refinement
        array_audio = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.sample_width == 2:  # 16-bit audio
            array_audio = array_audio / 32768.0
        
        # Calculate energy profile of the entire audio
        energy = np.square(array_audio)
        # Smooth the energy profile
        window_size = int(0.02 * sample_rate)  # 20ms window
        if window_size > 1:
            smoothed_energy = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
        else:
            smoothed_energy = energy
        
        # Process target words
        timestamps = []
        for i, word_idx in enumerate(target_indices):
            if 0 <= word_idx < len(positions):
                word = target_words[i]
                word_start, word_end = positions[word_idx]
                
                # Apply context-aware padding
                word_length = len(word)
                context_factor = min(1.2, max(0.8, word_length / 5))
                actual_padding = int(padding_ms * context_factor)
                
                # Apply padding with pre-emptive adjustment
                padded_start = max(0, word_start - actual_padding - 30)  # Start earlier
                padded_end = min(total_duration_ms, word_end + actual_padding)
                
                # Convert time to samples for energy analysis
                start_sample = int((padded_start / 1000) * sample_rate)
                end_sample = int((padded_end / 1000) * sample_rate)
                
                # Ensure valid indices
                start_sample = max(0, min(start_sample, len(smoothed_energy) - 1))
                end_sample = max(start_sample + 1, min(end_sample, len(smoothed_energy)))
                
                # Extract segment energy
                segment_energy = smoothed_energy[start_sample:end_sample]
                
                # Analyze energy to find true speech boundaries
                if len(segment_energy) > 0:
                    # Threshold at percentage of max energy
                    threshold = 0.15 * np.max(segment_energy)
                    speech_positions = np.where(segment_energy > threshold)[0]
                    
                    if len(speech_positions) > 0:
                        # Find first and last positions above threshold
                        speech_start = speech_positions[0]
                        speech_end = speech_positions[-1]
                        
                        # Convert back to ms and apply pre-emptive adjustment
                        refined_start = padded_start + int((speech_start / len(segment_energy)) * (padded_end - padded_start)) - 20
                        refined_end = padded_start + int((speech_end / len(segment_energy)) * (padded_end - padded_start)) + 10
                        
                        # Ensure valid boundaries
                        refined_start = max(0, refined_start)
                        refined_end = min(total_duration_ms, refined_end)
                        
                        timestamps.append((refined_start, refined_end, word))
                        continue
                
                # Fallback to adjusted boundaries
                timestamps.append((padded_start, padded_end, word))
        
        return timestamps
    except Exception as e:
        logger.error(f"Error in improved forced alignment: {e}", exc_info=True)
        return []

def improved_word_timestamp_estimation(audio_path, all_words, target_words, target_indices, audio_duration, padding_ms=100):
    """Improved timestamp estimation using energy analysis and pre-emptive timing"""
    try:
        logger.info("Using improved estimated timestamps with energy analysis")
        
        # Load audio for analysis
        audio = AudioSegment.from_wav(audio_path)
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        
        # Calculate energy profile
        frame_size = int(0.02 * sample_rate)  # 20ms frames
        hop_size = int(0.01 * sample_rate)    # 10ms hop
        
        # Process audio in frames to get energy profile
        num_frames = 1 + (len(samples) - frame_size) // hop_size
        energy_profile = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_size
            if end <= len(samples):
                frame = samples[start:end]
                energy_profile[i] = np.sum(frame**2)
        
        # Normalize energy profile
        if np.max(energy_profile) > 0:
            energy_profile = energy_profile / np.max(energy_profile)
        
        # Detect speech segments (above energy threshold)
        threshold = 0.15
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        for i, energy in enumerate(energy_profile):
            time_ms = int((i * hop_size / sample_rate) * 1000)
            
            if not in_speech and energy > threshold:
                in_speech = True
                speech_start = time_ms
            elif in_speech and energy < threshold:
                in_speech = False
                speech_segments.append((speech_start, time_ms))
        
        # Handle the case if we end in speech
        if in_speech:
            speech_segments.append((speech_start, int(audio_duration * 1000)))
        
        # If no speech segments detected, fallback to simple estimation
        if not speech_segments:
            # Simple time-based estimation
            word_duration = (audio_duration * 1000) / len(all_words)
            timestamps = []
            for i, idx in enumerate(target_indices):
                start_time = max(0, int(idx * word_duration - padding_ms))
                end_time = min(int(audio_duration * 1000), int((idx + 1) * word_duration + padding_ms))
                timestamps.append((start_time, end_time, target_words[i]))
            return timestamps
        
        # Distribute words among speech segments
        total_words = len(all_words)
        if total_words == 0:
            return []
            
        # Calculate total speech duration
        total_speech_duration = sum(end - start for start, end in speech_segments)
        if total_speech_duration == 0:
            total_speech_duration = audio_duration * 1000
        # Distribute words across speech segments
        words_per_ms = total_words / total_speech_duration
        word_positions = []
        word_idx = 0
        
        for start, end in speech_segments:
            segment_duration = end - start
            num_words_in_segment = int(segment_duration * words_per_ms)
            
            if num_words_in_segment > 0:
                segment_word_duration = segment_duration / num_words_in_segment
                for j in range(num_words_in_segment):
                    if word_idx < total_words:
                        word_start = start + int(j * segment_word_duration)
                        word_end = start + int((j + 1) * segment_word_duration)
                        word_positions.append((word_start, word_end))
                        word_idx += 1
        
        
        remaining_words = total_words - word_idx
        if remaining_words > 0 and speech_segments:
            last_segment_start, last_segment_end = speech_segments[-1]
            segment_duration = last_segment_end - last_segment_start
            word_duration = segment_duration / remaining_words
            
            for j in range(remaining_words):
                word_start = last_segment_start + int(j * word_duration)
                word_end = last_segment_start + int((j + 1) * word_duration)
                word_positions.append((word_start, word_end))
        
        # Extract target word timestamps
        timestamps = []
        for i, idx in enumerate(target_indices):
            if idx < len(word_positions):
                word_start, word_end = word_positions[idx]
                
                # Apply padding and pre-emptive adjustment
                word_start = max(0, word_start - padding_ms - 30)  # Start earlier
                word_end = min(int(audio_duration * 1000), word_end + padding_ms)
                
                timestamps.append((word_start, word_end, target_words[i]))
        
        return timestamps
    except Exception as e:
        logger.error(f"Error in improved timestamp estimation: {e}", exc_info=True)
        # Fallback to basic estimation
        word_duration = (audio_duration * 1000) / max(1, len(all_words))
        return [
            (max(0, int(idx * word_duration - padding_ms - 30)),
             min(int(audio_duration * 1000), int((idx + 1) * word_duration + padding_ms)),
             target_words[i])
            for i, idx in enumerate(target_indices)
        ]

def advanced_energy_based_refinement(audio_path, timestamps, padding_ms=100):
    """Advanced energy-based refinement of word boundaries for more precise censoring"""
    try:
        # Load audio for analysis
        audio = AudioSegment.from_wav(audio_path)
        audio_array = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        
        # Process each timestamp for refinement
        refined_timestamps = []
        
        for start_time, end_time, word in timestamps:
            # Convert ms to samples with expanded analysis window
            analysis_start_ms = max(0, start_time - 200)  # Look earlier
            analysis_end_ms = min(len(audio), end_time + 200)  # Look later
            
            analysis_start = int(analysis_start_ms * sample_rate / 1000)
            analysis_end = int(analysis_end_ms * sample_rate / 1000)
            
            # Ensure valid sample indices
            analysis_start = max(0, min(analysis_start, len(audio_array) - 1))
            analysis_end = max(analysis_start + 1, min(analysis_end, len(audio_array)))
            
            # Extract audio segment for analysis
            segment = audio_array[analysis_start:analysis_end]
            
            if len(segment) > 0:
                # Calculate energy profile
                energy = np.square(segment)
                
                # Apply spectral analysis for better speech detection
                window_size = int(0.025 * sample_rate)  # 25ms window
                
                # Ensure window size is valid
                if window_size > 1 and len(energy) > window_size:
                    # Apply smoothing to energy profile
                    smoothed_energy = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
                    
                    # Identify speech activity using dynamic thresholding
                    if np.max(smoothed_energy) > 0:
                        # Use percentage of max energy as threshold
                        threshold = 0.12 * np.max(smoothed_energy)
                        
                        # Find speech activity regions
                        speech_indices = np.where(smoothed_energy > threshold)[0]
                        
                        if len(speech_indices) > 0:
                            
                            gaps = np.where(np.diff(speech_indices) > window_size)[0]
                            segments = []
                            
                            if len(gaps) > 0:
                                
                                prev = 0
                                for gap in gaps:
                                    segments.append((speech_indices[prev], speech_indices[gap]))
                                    prev = gap + 1
                                
                                # Add the last segment
                                if prev < len(speech_indices):
                                    segments.append((speech_indices[prev], speech_indices[-1]))
                            else:
                                
                                segments = [(speech_indices[0], speech_indices[-1])]
                            
                            
                            expected_start_sample = int((start_time - analysis_start_ms) * sample_rate / 1000)
                            expected_end_sample = int((end_time - analysis_start_ms) * sample_rate / 1000)
                            expected_center = (expected_start_sample + expected_end_sample) / 2
                            
                            best_segment = None
                            best_score = float('inf')
                            
                            for seg_start, seg_end in segments:
                                
                                seg_center = (seg_start + seg_end) / 2
                                distance = abs(seg_center - expected_center)
                                
                                if distance < best_score:
                                    best_score = distance
                                    best_segment = (seg_start, seg_end)
                            
                            if best_segment:
                                
                                seg_start, seg_end = best_segment
                                refined_start = analysis_start_ms + int(seg_start * 1000 / sample_rate)
                                refined_end = analysis_start_ms + int(seg_end * 1000 / sample_rate)
                                
                                
                                refined_start = max(0, refined_start - 40)  # Start earlier for better censoring
                                refined_end = refined_end + 10  # Small padding at the end
                                
                                refined_timestamps.append((refined_start, refined_end, word))
                                continue
            
            refined_start = max(0, start_time - 30)  # Start slightly earlier
            refined_end = end_time + 10
            refined_timestamps.append((refined_start, refined_end, word))
        
        return refined_timestamps
    except Exception as e:
        logger.error(f"Error in advanced energy-based refinement: {e}", exc_info=True)
        return timestamps  # Return original timestamps on failure

def apply_preemptive_timing(timestamps, pre_ms=25):
    """Apply pre-emptive timing adjustment to ensure censoring starts before the word"""
    return [(max(0, start - pre_ms), end, word) for start, end, word in timestamps]

def enhanced_audio_censoring(input_path, output_path, timestamps, censor_type='beep'):
    """Enhanced audio censoring with smoother transitions and better timing"""
    try:
        original_audio = AudioSegment.from_wav(input_path)
        original_length = len(original_audio)
        
        # Sort and merge timestamps to reduce processing
        timestamps_sorted = sorted(timestamps, key=lambda x: x[0])
        merged_timestamps = []
        
        # Efficiently merge overlapping timestamps
        for start_time, end_time, word in timestamps_sorted:
            start_time = max(0, min(start_time, original_length))
            end_time = max(0, min(end_time, original_length))
            
            if start_time >= end_time:
                continue
                
            if merged_timestamps and start_time <= merged_timestamps[-1][1]:
                merged_timestamps[-1] = (
                    merged_timestamps[-1][0],
                    max(merged_timestamps[-1][1], end_time),
                    f"{merged_timestamps[-1][2]}, {word}"
                )
            else:
                merged_timestamps.append((start_time, end_time, word))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            segment_data = []
            if merged_timestamps and merged_timestamps[0][0] > 0:
                segment_data.append((0, merged_timestamps[0][0], None))
            for i, (start_time, end_time, word) in enumerate(merged_timestamps):
                segment_data.append((start_time, end_time, word))
                if i < len(merged_timestamps) - 1 and end_time < merged_timestamps[i+1][0]:
                    segment_data.append((end_time, merged_timestamps[i+1][0], None))
            if merged_timestamps and merged_timestamps[-1][1] < original_length:
                segment_data.append((merged_timestamps[-1][1], original_length, None))
            if not merged_timestamps:
                segment_data = [(0, original_length, None)]
            for start, end, word in segment_data:
                if word is None:
                    futures.append(executor.submit(lambda s, e: original_audio[s:e], start, end))
                else:
                    futures.append(executor.submit(
                        lambda s, e, w: generate_enhanced_censor_segment(original_audio, s, e, w, censor_type),
                        start, end, word
                    ))
            
            segments = [future.result() for future in futures]
        

        censored_audio = None
        
        if segments:

            censored_audio = segments[0]
            for i in range(1, len(segments)):
                prev_segment = segments[i-1]
                curr_segment = segments[i]
                crossfade_duration = 10  # 10ms crossfade
                if len(prev_segment) >= crossfade_duration and len(curr_segment) >= crossfade_duration:
                    censored_audio = censored_audio.append(curr_segment, crossfade=crossfade_duration)
                else:
                    censored_audio = censored_audio + curr_segment
        else:
            censored_audio = original_audio
        

        if len(censored_audio) != original_length:
            if len(censored_audio) > original_length:
                censored_audio = censored_audio[:original_length]
            else:
                censored_audio = censored_audio + AudioSegment.silent(duration=original_length - len(censored_audio))
        

        censored_audio.export(output_path, format="wav", parameters=["-q:a", "0"])
        logger.info(f"Created censored audio file: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error in enhanced audio censoring: {e}", exc_info=True)
        return False

def generate_enhanced_censor_segment(original_audio, start_time, end_time, word, censor_type):
    """Generate improved censored segments with better sound quality and transitions"""
    segment_duration = end_time - start_time
    
    if censor_type == 'beep':
        beep = generate_beep(segment_duration)
        fade_duration = min(15, segment_duration // 8)  
        beep = beep.fade_in(fade_duration).fade_out(fade_duration)
        context_before = original_audio[max(0, start_time-300):start_time]
        context_after = original_audio[end_time:min(len(original_audio), end_time+300)]
        target_dbfs = None
        if len(context_before) > 0:
            before_dbfs = context_before.dBFS
            if before_dbfs > -float('inf'):
                target_dbfs = before_dbfs
        
        if target_dbfs is None and len(context_after) > 0:
            after_dbfs = context_after.dBFS
            if after_dbfs > -float('inf'):
                target_dbfs = after_dbfs
        if target_dbfs is not None:
            beep = beep.apply_gain(target_dbfs - beep.dBFS + 3)
        if len(beep) != segment_duration:
            if len(beep) > segment_duration:
                beep = beep[:segment_duration]
            else:
                beep = beep + AudioSegment.silent(duration=segment_duration - len(beep))
        
        logger.info(f"Enhanced beep for word(s) '{word}' at {start_time}-{end_time}ms")
        return beep
    else:
        segment = original_audio[start_time:end_time]
        ambient_level = -45  
        ambient_noise = segment.apply_gain(ambient_level - segment.dBFS if segment.dBFS > -float('inf') else -80)
        fade_duration = min(20, segment_duration // 6)
        ambient_noise = ambient_noise.fade_in(fade_duration).fade_out(fade_duration)
        
        logger.info(f"Replaced word(s) '{word}' with improved silence at {start_time}-{end_time}ms")
        return ambient_noise

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.abspath(UPLOAD_FOLDER), filename)

if __name__ == '__main__':
    app.run(debug=False, threaded=True)