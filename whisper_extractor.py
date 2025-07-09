#!/usr/bin/env python3
"""
Smart Whisper Mention Extractor

Efficient approach:
1. Use existing SRT files to find approximate locations (fast)
2. Extract only those small segments 
3. Run Whisper on small segments for precise timing (fast)
4. Create compilation with exact timing

This is much faster than running Whisper on entire long videos!
"""

import os
import re
import subprocess
import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

try:
    import whisper
except ImportError:
    print("Please install: pip install openai-whisper")
    exit(1)


def parse_srt_timestamp(timestamp_str):
    """Convert SRT timestamp format to seconds"""
    time_part, ms_part = timestamp_str.split(',')
    hours, minutes, seconds = map(int, time_part.split(':'))
    milliseconds = int(ms_part)
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def find_srt_matches(srt_file, phrases, padding=2.0):
    """Find approximate matches in SRT file"""
    matches = []
    
    # Find corresponding video file
    srt_path = Path(srt_file)
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov']
    video_file = None
    
    if srt_path.name.endswith('.en.srt'):
        base_name = srt_path.name[:-7]  # Remove '.en.srt'
        for ext in video_extensions:
            potential_video = srt_path.parent / (base_name + ext)
            if potential_video.exists():
                video_file = str(potential_video)
                break
    else:
        for ext in video_extensions:
            potential_video = srt_path.with_suffix(ext)
            if potential_video.exists():
                video_file = str(potential_video)
                break
    
    if not video_file:
        return matches
    
    try:
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return matches
    
    # Parse SRT blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3 or not lines[0].strip().isdigit() or '-->' not in lines[1]:
            continue
            
        timestamp_line = lines[1].strip()
        subtitle_text = ' '.join(lines[2:]).strip()
        
        # Check for phrase matches
        for phrase in phrases:
            if phrase.lower() in subtitle_text.lower():
                try:
                    start_str, end_str = timestamp_line.split(' --> ')
                    start_time = parse_srt_timestamp(start_str.strip())
                    end_time = parse_srt_timestamp(end_str.strip())
                    
                    # Add padding for context
                    segment_start = max(0, start_time - padding)
                    segment_end = end_time + padding
                    
                    matches.append({
                        'video_file': video_file,
                        'phrase': phrase,
                        'subtitle_text': subtitle_text,
                        'segment_start': segment_start,
                        'segment_end': segment_end,
                        'original_start': start_time,
                        'original_end': end_time
                    })
                except:
                    continue
    
    return matches


def extract_audio_segment(video_file, start_time, end_time, output_file):
    """Extract audio segment for Whisper analysis"""
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', video_file,
        '-t', str(end_time - start_time),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        output_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def find_precise_timing_with_whisper(audio_file, phrase, segment_start, model):
    """Use Whisper to find precise timing within the small segment"""
    try:
        result = model.transcribe(audio_file)
        segments = result.get('segments', [])
        
        best_match = None
        best_confidence = 0
        
        for segment in segments:
            segment_text = segment.get('text', '').strip()
            if phrase.lower() in segment_text.lower():
                # Calculate confidence based on how well the phrase matches
                confidence = len(phrase) / len(segment_text) if segment_text else 0
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    # Convert segment timing back to video timing
                    precise_start = segment_start + segment.get('start', 0)
                    precise_end = segment_start + segment.get('end', segment.get('start', 0) + 3)
                    
                    best_match = {
                        'precise_start': precise_start,
                        'precise_end': precise_end,
                        'confidence': confidence,
                        'whisper_text': segment_text
                    }
        
        return best_match
    except:
        return None


def create_final_clip(video_file, start_time, end_time, phrase, clip_id, output_file):
    """Create final video clip with overlay"""
    duration = end_time - start_time
    counter_text = f"'{phrase} Counter: {clip_id}'".replace(':', '\\:')
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', video_file,
        '-t', str(duration)
    ]
    
    # Add ding if available
    ding_file = Path('audio/ding.mp3')
    if ding_file.exists():
        cmd.extend(['-i', str(ding_file)])
    
    # Add font overlay if available
    font_file = Path('fonts/CoolveticaRg.otf')
    if font_file.exists():
        font_path = str(font_file).replace('\\', '/')
        text_filter = f"drawtext=fontfile='{font_path}':text={counter_text}:fontsize=48:fontcolor=white:x=w-tw-20:y=20:shadowcolor=black:shadowx=2:shadowy=2"
        
        if ding_file.exists():
            cmd.extend(['-filter_complex', f"[0:v]{text_filter}[v];[0:a][1:a]amix=inputs=2:duration=first[a]"])
            cmd.extend(['-map', '[v]', '-map', '[a]'])
        else:
            cmd.extend(['-vf', text_filter])
    elif ding_file.exists():
        cmd.extend(['-filter_complex', "[0:a][1:a]amix=inputs=2:duration=first[a]"])
        cmd.extend(['-map', '0:v', '-map', '[a]'])
    
    cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', str(output_file)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Smart Whisper Mention Extractor')
    parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large'])
    parser.add_argument('--test', action='store_true', help='Save individual clips')
    parser.add_argument('phrases', nargs='*', help='Phrases to find')
    
    args = parser.parse_args()
    
    # Get phrases
    if args.phrases:
        phrases = args.phrases
    else:
        phrases = []
        print("Enter phrases (empty line to finish):")
        while True:
            phrase = input("Phrase: ").strip()
            if not phrase:
                break
            phrases.append(phrase)
    
    if not phrases:
        print("No phrases provided")
        return
    
    print("=== Smart Whisper Mention Extractor ===")
    print(f"🎯 Phrases: {phrases}")
    
    # Check setup
    videos_dir = Path('videos')
    if not videos_dir.exists():
        print(f"❌ {videos_dir} folder not found")
        return
    
    # Find SRT files
    srt_files = list(videos_dir.glob('*.srt'))
    if not srt_files:
        print("❌ No SRT files found in videos/ folder")
        return
    
    print(f"📄 Found {len(srt_files)} SRT files")
    
    # Load Whisper model
    print(f"🔄 Loading Whisper {args.model} model...")
    model = whisper.load_model(args.model)
    print("✅ Model loaded")
    
    # Find all SRT matches first (fast)
    print("\n🔍 Scanning SRT files for approximate matches...")
    all_matches = []
    for srt_file in srt_files:
        matches = find_srt_matches(srt_file, phrases)
        all_matches.extend(matches)
        if matches:
            print(f"  📄 {srt_file.name}: {len(matches)} matches")
    
    if not all_matches:
        print("❌ No matches found in SRT files")
        return
    
    print(f"\n✅ Found {len(all_matches)} total matches in SRT files")
    
    # Now process each match with Whisper for precision
    print("🎯 Analyzing matches with Whisper for precise timing...")
    
    final_clips = []
    clip_counter = 0
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    if args.test:
        clips_dir = output_dir / 'clips'
        clips_dir.mkdir(exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        
        for i, match in enumerate(all_matches):
            print(f"\n  🎬 Processing match {i+1}/{len(all_matches)}: '{match['phrase']}'")
            
            # Extract small audio segment
            audio_file = temp_dir / f"segment_{i}.wav"
            if not extract_audio_segment(match['video_file'], match['segment_start'], 
                                       match['segment_end'], str(audio_file)):
                print(f"    ❌ Failed to extract audio segment")
                continue
            
            # Find precise timing with Whisper
            precise_match = find_precise_timing_with_whisper(
                str(audio_file), match['phrase'], match['segment_start'], model
            )
            
            if not precise_match:
                print(f"    ⚠️  Whisper couldn't find precise timing, using SRT timing")
                # Fallback to SRT timing
                precise_start = match['original_start'] - 0.5
                precise_end = match['original_end'] + 0.5
            else:
                print(f"    ✅ Precise timing found (confidence: {precise_match['confidence']:.2f})")
                precise_start = max(0, precise_match['precise_start'] - 0.3)
                precise_end = precise_match['precise_end'] + 0.3
            
            # Create final clip
            clip_counter += 1
            clip_file = temp_dir / f"final_clip_{clip_counter:04d}.mp4"
            
            if create_final_clip(match['video_file'], precise_start, precise_end, 
                               match['phrase'], clip_counter, str(clip_file)):
                
                if args.test:
                    # Save test clip
                    test_name = f"clip_{clip_counter:04d}_{match['phrase'].replace(' ', '_')}_whisper.mp4"
                    test_path = clips_dir / test_name
                    import shutil
                    shutil.copy2(str(clip_file), str(test_path))
                    print(f"    🧪 Test clip saved: {test_name}")
                
                final_clips.append(str(clip_file))
                print(f"    ✅ Clip {clip_counter} created")
            else:
                print(f"    ❌ Failed to create final clip")
        
        if not final_clips:
            print("❌ No clips were successfully created")
            return
        
        # Concatenate all clips
        print(f"\n🎥 Creating compilation from {len(final_clips)} clips...")
        
        # Create concat file
        concat_file = temp_dir / 'concat.txt'
        with open(concat_file, 'w') as f:
            for clip_file in final_clips:
                f.write(f"file '{clip_file.replace(chr(92), '/')}'\n")
        
        # Generate output filename
        safe_phrases = '_'.join(re.sub(r'[^\w]', '', p) for p in phrases[:2])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"whisper_{safe_phrases}_{timestamp}.mp4"
        
        # Concatenate
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat_file),
            '-c', 'copy', str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Fallback to re-encoding
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat_file),
                '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac', str(output_file)
            ]
            subprocess.run(cmd, capture_output=True, text=True)
        
        # Save log
        log_file = output_file.with_suffix('.json')
        log_data = {
            'compilation_file': str(output_file),
            'timestamp': datetime.now().isoformat(),
            'method': 'smart_whisper',
            'model': args.model,
            'summary': {
                'total_clips': len(final_clips),
                'phrases_searched': phrases,
                'srt_matches_found': len(all_matches),
                'successful_clips': len(final_clips)
            }
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✅ Done! Created: {output_file}")
        print(f"📊 {len(final_clips)} clips | 📝 Log: {log_file}")
        
        if args.test:
            print(f"🧪 Individual clips saved in: {clips_dir}")


if __name__ == "__main__":
    main() 