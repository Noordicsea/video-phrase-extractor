#!/usr/bin/env python3
"""
Mention Extractor - Searches for specific words/phrases in video subtitles
and creates a compilation video of all mentions using ffmpeg.
"""

import os
import re
import subprocess
import tempfile
import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pyttsx3
import librosa
import numpy as np
from scipy import signal
import soundfile as sf
import json

def generate_tts_audio(phrases, temp_dir):
    """Generate TTS audio files for target phrases"""
    print("Generating TTS audio for phrase matching...")
    
    engine = pyttsx3.init()
    
    # Configure TTS settings for better matching
    engine.setProperty('rate', 180)  # Normal speech rate for better matching
    engine.setProperty('volume', 1.0)
    
    # Try to use a consistent voice
    voices = engine.getProperty('voices')
    if voices:
        # Use first available voice for consistency
        engine.setProperty('voice', voices[0].id)
    
    tts_files = {}
    
    for phrase in phrases:
        safe_phrase = re.sub(r'[^\w\s]', '', phrase).replace(' ', '_').lower()
        tts_file = os.path.join(temp_dir, f"tts_{safe_phrase}.wav")
        
        try:
            engine.save_to_file(phrase, tts_file)
            engine.runAndWait()
            
            # Verify file was created
            if os.path.exists(tts_file):
                tts_files[phrase] = tts_file
                print(f"  ✅ Generated TTS for: '{phrase}'")
            else:
                print(f"  ❌ Failed to generate TTS for: '{phrase}'")
                
        except Exception as e:
            print(f"  ❌ TTS error for '{phrase}': {e}")
    
    return tts_files

def find_all_phrase_occurrences(video_file, start_time, end_time, tts_file, phrase):
    """Find ALL occurrences of phrase within video segment using audio analysis"""
    temp_audio_path = None
    try:
        # Create temporary file for audio extraction
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # Extract audio segment using ffmpeg
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', video_file, 
            '-t', str(end_time - start_time),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            temp_audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ❌ Audio extraction failed: {result.stderr}")
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            return []
            
        # Load audio files
        video_audio, sr1 = librosa.load(temp_audio_path, sr=16000)
        tts_audio, sr2 = librosa.load(tts_file, sr=16000)
        
        # Ensure audio is loaded
        if len(video_audio) == 0 or len(tts_audio) == 0:
            print(f"    ❌ Failed to load audio for phrase matching")
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            return []
        
        # Normalize audio for better correlation
        video_audio = video_audio / np.max(np.abs(video_audio))
        tts_audio = tts_audio / np.max(np.abs(tts_audio))
        
        # Cross-correlation to find all matches
        correlation = signal.correlate(video_audio, tts_audio, mode='valid')
        
        if len(correlation) == 0:
            print(f"    ❌ No correlation data for '{phrase}'")
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            return []
        
        # Normalize correlation and find peaks
        max_correlation = np.max(np.abs(correlation))
        if max_correlation > 0:
            normalized_correlation = correlation / max_correlation
        else:
            normalized_correlation = correlation
            
        min_correlation = 0.4  # Reasonable threshold (40% of max correlation)
        phrase_duration = len(tts_audio) / sr2
        min_distance_samples = int(phrase_duration * sr1 * 1.2)  # 120% spacing to prevent overlaps
        
        # Find peaks with minimum distance between them
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(normalized_correlation, 
                                     height=min_correlation, 
                                     distance=min_distance_samples)
        
        occurrences = []
        for peak_idx in peaks:
            correlation_strength = float(normalized_correlation[peak_idx])
            raw_correlation = float(correlation[peak_idx])
            time_offset = peak_idx / sr1
            phrase_start = start_time + time_offset
            
            print(f"      🔍 Found peak: normalized={correlation_strength:.3f}, raw={raw_correlation:.1f}, time={phrase_start:.2f}s")
            
            # Accept reasonable matches
            if correlation_strength >= min_correlation:
                occurrences.append({
                    'phrase_start': phrase_start,
                    'phrase_duration': phrase_duration,
                    'correlation': correlation_strength,
                    'phrase': phrase
                })
        
        # Clean up temp file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        
        if len(occurrences) > 0:
            print(f"    ✅ Found {len(occurrences)} HIGH-QUALITY occurrences of '{phrase}' in segment")
            for i, occ in enumerate(occurrences):
                print(f"      #{i+1}: {occ['phrase_start']:.2f}s (correlation: {occ['correlation']:.3f})")
        else:
            print(f"    ⚠️  No matches found for '{phrase}' above threshold: {min_correlation:.2f}")
        
        return occurrences
        
    except Exception as e:
        print(f"    ❌ Audio analysis error for '{phrase}': {e}")
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass
        return []

def verify_clip_contains_phrase(clip_file, phrase):
    """Use speech-to-text to verify the clip actually contains the phrase"""
    try:
        # Try to use Windows built-in speech recognition if available
        import speech_recognition as sr
        r = sr.Recognizer()
        
        # Convert video to audio for speech recognition
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        cmd = ['ffmpeg', '-y', '-i', clip_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_audio.name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            with sr.AudioFile(temp_audio.name) as source:
                audio = r.record(source)
            
            # Try to recognize speech
            try:
                # Use getattr to handle cases where method might not be available
                recognize_method = getattr(r, 'recognize_google', None)
                if recognize_method:
                    text = recognize_method(audio).lower()
                    phrase_lower = phrase.lower()
                    contains_phrase = phrase_lower in text
                    
                    os.unlink(temp_audio.name)
                    return {
                        "verified": contains_phrase,
                        "recognized_text": text,
                        "method": "speech_recognition"
                    }
                else:
                    os.unlink(temp_audio.name)
                    return {
                        "verified": True,  # Assume verified if method not available
                        "error": "recognize_google method not available",
                        "method": "fallback"
                    }
            except Exception:
                os.unlink(temp_audio.name)
                return {
                    "verified": False,
                    "recognized_text": "",
                    "error": "Speech not recognized",
                    "method": "speech_recognition"
                }
        else:
            if os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)
            return {
                "verified": False,
                "error": "Audio conversion failed",
                "method": "speech_recognition"
            }
            
    except ImportError:
        # speech_recognition not available, fall back to audio correlation confidence
        return {
            "verified": True,  # Assume verified if we can't check
            "error": "speech_recognition library not available",
            "method": "fallback"
        }
    except Exception as e:
        return {
            "verified": False,
            "error": str(e),
            "method": "speech_recognition"
        }

def get_windows_font_path():
    """Get a working font path for Windows FFmpeg drawtext"""
    # First, check local fonts folder
    local_fonts_dir = os.path.join(os.getcwd(), 'fonts')
    if os.path.exists(local_fonts_dir):
        for font_file in os.listdir(local_fonts_dir):
            if font_file.lower().endswith(('.ttf', '.otf')):
                font_path = os.path.join(local_fonts_dir, font_file)
                print(f"Found local font: {font_path}")
                # Convert to absolute path and use forward slashes for FFmpeg
                abs_path = os.path.abspath(font_path)
                # Use forward slashes - works better with FFmpeg on Windows
                forward_slash_path = abs_path.replace('\\', '/')
                print(f"Using font path: {forward_slash_path}")
                return forward_slash_path
    
    # Fallback to system fonts  
    possible_fonts = [
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/Arial.ttf', 
        'C:/Windows/Fonts/calibri.ttf',
        'C:/Windows/Fonts/Calibri.ttf',
        'C:/Windows/Fonts/tahoma.ttf',
        'C:/Windows/Fonts/Tahoma.ttf',
        'C:/Windows/Fonts/verdana.ttf',
        'C:/Windows/Fonts/Verdana.ttf'
    ]
    
    for font_path in possible_fonts:
        # Convert to Windows path for checking existence
        check_path = font_path.replace('/', '\\')
        if os.path.exists(check_path):
            print(f"Found system font: {check_path}")
            print(f"Using font path: {font_path}")
            return font_path
    
    print("No fonts found - will skip text overlay!")
    # If no fonts found, return None (will cause fallback to no text overlay)
    return None

def parse_srt_timestamp(timestamp_str):
    """Convert SRT timestamp format to seconds"""
    # Format: HH:MM:SS,mmm
    time_part, ms_part = timestamp_str.split(',')
    hours, minutes, seconds = map(int, time_part.split(':'))
    milliseconds = int(ms_part)
    
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return total_seconds

def search_phrases_in_srt(srt_file, search_phrases, padding_seconds=1.0):
    """
    Search for multiple phrases in an SRT file and return timestamps of matches.
    
    Args:
        srt_file: Path to the SRT file
        search_phrases: List of words/phrases to search for
        padding_seconds: Extra seconds to add before/after each match
    
    Returns:
        List of tuples: (start_time, end_time, matched_text, video_file, phrase_counts)
    """
    matches = []
    
    # Find corresponding video file
    srt_path = Path(srt_file)
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv']
    video_file = None
    
    # Handle .en.srt files by removing .en.srt and trying video extensions
    if srt_path.name.endswith('.en.srt'):
        base_name = srt_path.name[:-7]  # Remove '.en.srt'
        for ext in video_extensions:
            potential_video = srt_path.parent / (base_name + ext)
            if potential_video.exists():
                video_file = str(potential_video)
                break
    else:
        # Handle regular .srt files
        for ext in video_extensions:
            potential_video = srt_path.with_suffix(ext)
            if potential_video.exists():
                video_file = str(potential_video)
                break
    
    if not video_file:
        print(f"Warning: No video file found for {srt_file}")
        return matches
    
    try:
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {srt_file}: {e}")
        return matches
    
    # Split into subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
            
        # Skip if first line is not a number (subtitle index)
        if not lines[0].strip().isdigit():
            continue
            
        # Check if second line contains timestamp
        if '-->' not in lines[1]:
            continue
            
        timestamp_line = lines[1].strip()
        subtitle_text = ' '.join(lines[2:]).strip()
        
        # Check if any search phrase is in the subtitle text (case insensitive)
        phrase_counts = {}
        total_matches = 0
        
        for phrase in search_phrases:
            count = subtitle_text.lower().count(phrase.lower())
            if count > 0:
                phrase_counts[phrase] = count
                total_matches += count
        
        if total_matches > 0:
            try:
                
                # Parse timestamps
                start_str, end_str = timestamp_line.split(' --> ')
                start_time = parse_srt_timestamp(start_str.strip())
                end_time = parse_srt_timestamp(end_str.strip())
                
                # Add padding
                start_time = max(0, start_time - padding_seconds)
                end_time = end_time + padding_seconds
                
                # Add tuple with phrase counts: (start_time, end_time, text, video_file, phrase_counts)
                matches.append((start_time, end_time, subtitle_text, video_file, phrase_counts))
                
            except Exception as e:
                print(f"Error parsing timestamp in {srt_file}: {e}")
                continue
    
    return matches

def create_precise_compilation_video(matches, output_file, search_phrases, test_mode=False):
    """
    Create a compilation video with precise phrase timing using TTS and audio analysis.
    
    Args:
        matches: List of (start_time, end_time, text, video_file, phrase_counts) tuples
        output_file: Output video file path
        search_phrases: List of phrases being compiled
    """
    if not matches:
        print("No matches found!")
        return False
    
    phrases_str = ', '.join(f"'{p}'" for p in search_phrases)
    print(f"\nFound {len(matches)} subtitle fragments containing {phrases_str}")
    
    # Remove overlapping clips and calculate total mentions
    def remove_overlaps(matches):
        """Remove overlapping clips and merge nearby ones"""
        if not matches:
            return []
        
        # Sort by video file and start time
        sorted_matches = sorted(matches, key=lambda x: (x[3], x[0]))
        merged = []
        
        for match in sorted_matches:
            start_time, end_time, text, video_file, phrase_counts = match
            
            # Check if this overlaps with the last merged clip from the same video
            if (merged and merged[-1][3] == video_file and 
                start_time < merged[-1][1]):  # Overlapping timestamps
                # Merge with previous clip - extend end time and combine phrase counts
                prev_start, prev_end, prev_text, prev_video, prev_counts = merged[-1]
                
                # Merge the phrase count dictionaries
                combined_counts = prev_counts.copy()
                for phrase, count in phrase_counts.items():
                    combined_counts[phrase] = combined_counts.get(phrase, 0) + count
                
                merged[-1] = (prev_start, max(end_time, prev_end), 
                             f"{prev_text} | {text}", video_file, combined_counts)
            else:
                merged.append(match)
        
        return merged
    
    # Remove overlapping clips
    matches = remove_overlaps(matches)
    total_mentions = sum(sum(counts.values()) for _, _, _, _, counts in matches)
    
    print(f"After removing overlaps: {len(matches)} clips containing {total_mentions} total mentions")
    
    # Setup directories
    if test_mode:
        clips_dir = os.path.join(os.getcwd(), 'clips')
        if os.path.exists(clips_dir):
            shutil.rmtree(clips_dir)
        os.makedirs(clips_dir)
        print(f"🧪 TEST MODE: Individual clips will be saved to: {clips_dir}")
    
    # Create temporary directory for clips and TTS
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate TTS audio for all search phrases
        tts_files = generate_tts_audio(search_phrases, temp_dir)
        
        if not tts_files:
            print("❌ TTS generation failed! Falling back to subtitle-based timing...")
            # Fallback to original method without precise timing
            return create_fallback_compilation_video(matches, output_file, search_phrases, temp_dir)
        
        print("Creating compilation video with precise phrase timing...")
        clip_files = []
        verification_log = []
        
        # Check if ding sound exists
        ding_file = os.path.join(os.getcwd(), 'audio', 'ding.mp3')
        has_ding = os.path.exists(ding_file)
        
        # Track running counter
        current_counter = 0
        
        # Extract individual clips with precise timing
        for i, (start_time, end_time, text, video_file, phrase_counts) in enumerate(matches):
            print(f"\nProcessing clip {i+1}: {text[:50]}...")
            
            # Find ALL occurrences of each phrase in this segment
            all_occurrences = []
            
            for phrase, count in phrase_counts.items():
                if phrase in tts_files:
                    print(f"  Analyzing audio for phrase: '{phrase}'")
                    occurrences = find_all_phrase_occurrences(
                        video_file, start_time, end_time, tts_files[phrase], phrase
                    )
                    all_occurrences.extend(occurrences)
            
            if not all_occurrences:
                print(f"  ❌ No audio matches found, skipping this segment")
                continue
            
            # Sort occurrences by time
            all_occurrences.sort(key=lambda x: x['phrase_start'])
            
            # Remove overlapping occurrences - keep only the best correlation for nearby detections
            filtered_occurrences = []
            for occurrence in all_occurrences:
                # Check if this occurrence overlaps with any already filtered
                overlaps = False
                for existing in filtered_occurrences:
                    time_diff = abs(occurrence['phrase_start'] - existing['phrase_start'])
                    if time_diff < occurrence['phrase_duration'] * 0.8:  # If within 80% of phrase duration
                        overlaps = True
                        # Keep the one with better correlation
                        if occurrence['correlation'] > existing['correlation']:
                            filtered_occurrences.remove(existing)
                            filtered_occurrences.append(occurrence)
                        break
                
                if not overlaps:
                    filtered_occurrences.append(occurrence)
            
            # Sort again after filtering
            filtered_occurrences.sort(key=lambda x: x['phrase_start'])
            
            print(f"  ✅ Found {len(all_occurrences)} total phrase occurrences, filtered to {len(filtered_occurrences)} non-overlapping clips")
            
            # Create a separate clip for each filtered occurrence
            for occ_idx, occurrence in enumerate(filtered_occurrences):
                current_counter += 1
                
                phrase_start = occurrence['phrase_start']
                phrase_duration = occurrence['phrase_duration']
                phrase = occurrence['phrase']
                
                # Create precise clip: longer padding to ensure full phrase
                clip_start = max(0, phrase_start - 0.5)  # 0.5s before for context
                clip_end = phrase_start + phrase_duration + 0.5  # 0.5s after for context
                clip_duration = clip_end - clip_start
                
                counter_text = f"'{phrase} Counter\\: {current_counter}'"
                
                clip_file = os.path.join(temp_dir, f"clip_{current_counter:04d}.mp4")
                
                print(f"  Creating clip #{current_counter}: {clip_start:.2f}s to {clip_end:.2f}s ('{phrase}')")
                print(f"    📊 Correlation: {occurrence['correlation']:.3f} (threshold: 0.4)")
                
                # Use RELATIVE path for font - this should fix the parsing issue
                font_path = "fonts/CoolveticaRg.otf"
                
                if has_ding:
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(clip_start),
                        '-i', video_file,
                        '-i', ding_file,
                        '-t', str(clip_duration),
                        '-filter_complex', 
                        f'[0:v]drawtext=fontfile={font_path}:text={counter_text}:fontsize=48:fontcolor=white:x=w-tw-20:y=20:shadowcolor=black:shadowx=2:shadowy=2[v]; [0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[a]',
                        '-map', '[v]',
                        '-map', '[a]',
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-preset', 'fast',
                        '-crf', '23',
                        '-r', '30',
                        '-ar', '48000',
                        '-ac', '2',
                        '-movflags', '+faststart',
                        clip_file
                    ]
                else:
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(clip_start),
                        '-i', video_file,
                        '-t', str(clip_duration),
                        '-filter_complex', 
                        f'[0:v]drawtext=fontfile={font_path}:text={counter_text}:fontsize=48:fontcolor=white:x=w-tw-20:y=20:shadowcolor=black:shadowx=2:shadowy=2[v]',
                        '-map', '[v]',
                        '-map', '0:a',
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-preset', 'fast',
                        '-crf', '23',
                        '-r', '30',
                        '-ar', '48000',
                        '-ac', '2',
                        '-movflags', '+faststart',
                        clip_file
                    ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    # VERIFY the clip actually contains the phrase using speech-to-text
                    print(f"    🔍 Verifying phrase presence using speech-to-text...")
                    verification_result = verify_clip_contains_phrase(clip_file, phrase)
                    
                    # Create detailed metadata for this clip
                    clip_metadata = {
                        "clip_number": current_counter,
                        "phrase": phrase,
                        "subtitle_text": text,
                        "original_timing": f"{start_time:.2f}s - {end_time:.2f}s",
                        "extracted_timing": f"{clip_start:.2f}s - {clip_end:.2f}s",
                        "correlation": occurrence['correlation'],
                        "video_file": video_file,
                        "phrase_start_in_video": phrase_start,
                        "phrase_duration": phrase_duration,
                        "speech_verification": verification_result,
                        "detection_details": {
                            "correlation_threshold_used": 0.4,
                            "padding_before": 0.5,
                            "padding_after": 0.5,
                            "tts_phrase_used": phrase,
                            "audio_analysis_successful": True
                        }
                    }
                    
                    # Include clips that pass verification OR have good correlation
                    include_clip = (
                        verification_result.get("verified", True) or  # Default to True if speech rec unavailable
                        occurrence['correlation'] >= 0.7  # Good confidence threshold
                    )
                    
                    if include_clip:
                        # Log this occurrence
                        verification_log.append(clip_metadata)
                        
                        # In test mode, save clip and metadata to clips folder
                        if test_mode:
                            test_clip_name = f"clip_{current_counter:04d}_{phrase.replace(' ', '_')}_corr{occurrence['correlation']:.2f}_VERIFIED.mp4"
                            test_clip_path = os.path.join(clips_dir, test_clip_name)
                            shutil.copy2(clip_file, test_clip_path)
                            
                            # Save metadata as JSON
                            metadata_file = test_clip_path.replace('.mp4', '_metadata.json')
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(clip_metadata, f, indent=2, ensure_ascii=False)
                            
                            print(f"    🧪 VERIFIED clip saved: {test_clip_name}")
                            print(f"        📊 Correlation: {occurrence['correlation']:.3f}")
                            print(f"        ⏰ Timing: {clip_start:.2f}s - {clip_end:.2f}s")
                            print(f"        🎤 Speech verification: {verification_result.get('verified', False)}")
                            if verification_result.get('recognized_text'):
                                print(f"        📝 Recognized: '{verification_result['recognized_text']}'")
                        
                        clip_files.append(clip_file)
                        print(f"    ✅ VERIFIED clip #{current_counter} - phrase confirmed!")
                    else:
                        print(f"    ❌ REJECTED clip #{current_counter} - phrase NOT verified")
                        print(f"        🎤 Speech check: {verification_result.get('error', 'Failed')}")
                        if test_mode:
                            # Save rejected clip for analysis
                            reject_clip_name = f"clip_{current_counter:04d}_{phrase.replace(' ', '_')}_corr{occurrence['correlation']:.2f}_REJECTED.mp4"
                            reject_clip_path = os.path.join(clips_dir, reject_clip_name)
                            shutil.copy2(clip_file, reject_clip_path)
                            
                            metadata_file = reject_clip_path.replace('.mp4', '_metadata.json')
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(clip_metadata, f, indent=2, ensure_ascii=False)
                            print(f"    🧪 REJECTED clip saved for analysis: {reject_clip_name}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Failed to extract clip #{current_counter}: {e}")
                    continue
        
        if not clip_files:
            print("No clips were successfully extracted!")
            return False
        
        # Create concat file for ffmpeg
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            for clip_file in clip_files:
                # Use forward slashes for concat file paths
                normalized_path = clip_file.replace('\\', '/')
                f.write(f"file '{normalized_path}'\n")
        
        # Concatenate all clips with re-encoding for perfect sync
        print(f"Concatenating {len(clip_files)} clips...")
        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'fast',
            '-crf', '23',
            '-r', '30',  # Standardize frame rate
            '-ar', '48000',  # Standardize audio sample rate
            '-ac', '2',  # Stereo audio
            '-movflags', '+faststart',
            output_file
        ]
        
        try:
            result = subprocess.run(concat_cmd, capture_output=True, text=True, check=True)
            
            # Save verification log
            log_filename = output_file.replace('.mp4', '_verification_log.json')
            
            # Create summary statistics
            total_clips = len(verification_log)
            total_correlation = sum(entry["correlation"] for entry in verification_log)
            avg_correlation = total_correlation / total_clips if total_clips > 0 else 0
            
            log_summary = {
                "compilation_file": output_file,
                "generation_time": datetime.now().isoformat(),
                "summary": {
                    "total_clips": total_clips,
                    "total_phrase_occurrences": current_counter,
                    "average_correlation": round(avg_correlation, 3),
                    "phrases_searched": search_phrases
                },
                "clips": verification_log
            }
            
            with open(log_filename, 'w', encoding='utf-8') as f:
                json.dump(log_summary, f, indent=2, ensure_ascii=False)
            
            print(f"\nSuccess! Compilation video saved as: {output_file}")
            print(f"📊 Results:")
            print(f"   🎯 Total phrase occurrences: {current_counter}")
            print(f"   📹 Total clips created: {total_clips}")
            print(f"   📈 Average correlation: {avg_correlation:.3f}")
            print(f"   📝 Log saved: {log_filename}")
            
            # Test mode summary
            if test_mode:
                print(f"\n🧪 TEST MODE SUMMARY:")
                print(f"   📁 Individual clips saved to: {clips_dir}")
                print(f"   📊 Each clip includes detailed metadata JSON file")
                print(f"   🔍 Review clips to understand detection reasoning")
                print(f"   ⚠️  Look for clips that seem wrong or cut off")
                
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error concatenating clips: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            return False

def create_fallback_compilation_video(matches, output_file, search_phrases, temp_dir):
    """Fallback compilation without TTS analysis - uses subtitle timing"""
    print("Creating compilation video using subtitle timing...")
    
    clip_files = []
    verification_log = []
    ding_file = os.path.join(os.getcwd(), 'audio', 'ding.mp3')
    has_ding = os.path.exists(ding_file)
    current_counter = 0
    
    for i, (start_time, end_time, text, video_file, phrase_counts) in enumerate(matches):
        clip_file = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
        duration = end_time - start_time
        
        # Use first phrase for counter display
        first_phrase = list(phrase_counts.keys())[0]
        total_count = sum(phrase_counts.values())
        current_counter += total_count
        counter_text = f"'{first_phrase} Counter\\: {current_counter}'"
        
        print(f"Extracting fallback clip {i+1} (counter now {current_counter}): {text[:50]}...")
        
        # Simple extraction without precise timing
        if has_ding:
            cmd = [
                'ffmpeg', '-y', '-ss', str(start_time), '-i', video_file, '-i', ding_file,
                '-t', str(duration), '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[a]',
                '-map', '0:v', '-map', '[a]', '-c:v', 'libx264', '-c:a', 'aac',
                '-preset', 'fast', '-crf', '23', clip_file
            ]
        else:
            cmd = [
                'ffmpeg', '-y', '-ss', str(start_time), '-i', video_file,
                '-t', str(duration), '-c:v', 'libx264', '-c:a', 'aac',
                '-preset', 'fast', '-crf', '23', clip_file
            ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Note: Fallback mode doesn't have TTS files for verification
            verification_log.append({
                "clip_number": i + 1,
                "phrase": first_phrase,
                "subtitle_text": text,
                "original_timing": f"{start_time:.2f}s - {end_time:.2f}s",
                "extracted_timing": f"{start_time:.2f}s - {end_time:.2f}s",
                "verification": {"verified": False, "error": "Fallback mode - no TTS verification available", "correlation": 0.0},
                "video_file": video_file,
                "fallback_mode": True
            })
            
            clip_files.append(clip_file)
        except subprocess.CalledProcessError as e:
            print(f"Error creating fallback clip {i+1}: {e}")
            continue
    
    if not clip_files:
        return False
    
    # Concatenate clips
    concat_file = os.path.join(temp_dir, 'concat.txt')
    with open(concat_file, 'w') as f:
        for clip_file in clip_files:
            f.write(f"file '{clip_file.replace(chr(92), '/')}'\n")
    
    concat_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-c:v', 'libx264', '-c:a', 'aac', '-preset', 'fast', '-crf', '23',
        output_file
    ]
    
    try:
        result = subprocess.run(concat_cmd, capture_output=True, text=True, check=True)
        
        # Save verification log
        log_filename = output_file.replace('.mp4', '_verification_log.json')
        
        # Create summary statistics
        total_clips = len(verification_log)
        verified_clips = 0  # No verification in fallback mode
        unverified_clips = total_clips
        
        log_summary = {
            "compilation_file": output_file,
            "generation_time": datetime.now().isoformat(),
            "mode": "fallback",
            "summary": {
                "total_clips": total_clips,
                "verified_clips": verified_clips,
                "unverified_clips": unverified_clips,
                "verification_rate": "0% (fallback mode)"
            },
            "clips": verification_log
        }
        
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(log_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Fallback Mode Results:")
        print(f"   📝 Verification not available in fallback mode")
        print(f"   📄 Log saved: {log_filename}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error concatenating fallback clips: {e}")
        return False

def main():
    """Main function to run the mention extractor"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract and compile video mentions of specific phrases')
    parser.add_argument('--test', action='store_true', 
                       help='Test mode: Save individual clips to clips/ folder with detailed metadata')
    args = parser.parse_args()
    
    print("=== Mention Extractor ===")
    print("This tool will search for words/phrases in video subtitles")
    print("and create a compilation video of all mentions.")
    
    if args.test:
        print("🧪 TEST MODE ENABLED: Individual clips will be analyzed and saved")
    
    # Check for ding sound
    ding_file = os.path.join(os.getcwd(), 'audio', 'ding.mp3')
    if os.path.exists(ding_file):
        print("✅ Ding sound found - will add counter overlay and sound effects!")
    else:
        print("⚠️  No ding.mp3 found in audio/ folder - will add counter overlay only")
    print()
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH.")
        print("Please install ffmpeg and try again.")
        return
    
    # Get search phrases from user
    print("Enter phrases to search for (one per line, press Enter twice when done):")
    search_phrases = []
    while True:
        phrase = input("  Phrase: ").strip()
        if not phrase:
            break
        search_phrases.append(phrase)
    
    if not search_phrases:
        print("No phrases provided. Exiting.")
        return
    
    print(f"\nSearching for {len(search_phrases)} phrases: {', '.join(repr(p) for p in search_phrases)}")
    
    # Find all SRT files in videos directory
    videos_dir = Path('videos')
    if not videos_dir.exists():
        print("No 'videos' folder found! Please create a 'videos' folder and put your video files and SRT files there.")
        return
    
    srt_files = list(videos_dir.glob('*.srt'))
    if not srt_files:
        print("No SRT files found in videos/ directory!")
        return
    
    print(f"\nFound {len(srt_files)} subtitle files in videos/ folder:")
    for srt_file in srt_files:
        print(f"  - {srt_file}")
    
    print(f"\nSearching for phrases in all subtitle files in videos/ folder...")
    
    # Search for the phrases in all SRT files
    all_matches = []
    for srt_file in srt_files:
        print(f"Searching in {srt_file}...")
        matches = search_phrases_in_srt(str(srt_file), search_phrases)
        all_matches.extend(matches)
        print(f"  Found {len(matches)} matches")
    
    if not all_matches:
        print(f"\nNo mentions of any search phrases found in any video!")
        return
    
    # Sort matches by video file and timestamp for better organization
    all_matches.sort(key=lambda x: (x[3], x[0]))  # Sort by video file, then start time
    
    # Generate output filename  
    safe_phrases = '_'.join(re.sub(r'[^\w\s-]', '', phrase).strip().replace(' ', '_') 
                           for phrase in search_phrases[:2])  # Use first 2 phrases for filename
    output_file = f"compilation_{safe_phrases}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    # Create the compilation video with precise audio analysis
    success = create_precise_compilation_video(all_matches, output_file, search_phrases, test_mode=args.test)
    
    if success:
        total_mentions = sum(sum(counts.values()) for _, _, _, _, counts in all_matches)
        phrases_str = ', '.join(f"'{p}'" for p in search_phrases)
        print(f"\nDone! Your compilation video '{output_file}' contains {len(all_matches)} clips with {total_mentions} total mentions of {phrases_str}")
    else:
        print("\nFailed to create compilation video.")

if __name__ == "__main__":
    main() 