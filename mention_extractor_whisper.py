#!/usr/bin/env python3
"""
Whisper Mention Extractor - v13 (Pulse Animation)

Implements a sophisticated three-stage text overlay animation:
1. Pre-mention: Semi-transparent text with no number.
2. The "Flash": A brief, fully-opaque pulse where the number appears.
3. Post-mention: Semi-transparent text WITH the number.
"""
import os
import re
import argparse
import json
import shutil
import tempfile
import random
from datetime import datetime
from pathlib import Path
import torch
import whisper
import ffmpeg

class WhisperSrtHybridExtractor:
    VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.webm'}
    WHISPER_MODEL_SIZE = "base"
    SRT_WINDOW_PADDING = 2.0
    CLIP_PADDING_BEFORE_S = (0.4, 1.0)
    CLIP_PADDING_AFTER_S  = (0.7, 1.5)
    DING_OFFSET_S = 0.1
    DEDUPLICATION_SECONDS = 2.0
    
    # <-- New setting for the flash duration
    FLASH_DURATION_S = 0.3
    
    FFMPEG_QUIET_MODE = True

    def __init__(self, search_phrases, test_mode=False, no_ding=False):
        self.search_phrases = [self._normalize_text(p) for p in search_phrases]
        self.test_mode = test_mode
        self.no_ding = no_ding
        self.videos_dir = Path('videos')
        self.output_dir = Path('output')
        self.clips_dir = self.output_dir / 'clips'
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        self.ding_file = Path('audio/ding.mp3').resolve()
        self.font_file = Path('fonts/CoolveticaRg.otf').resolve()
        self.has_ding = self.ding_file.exists() and not self.no_ding
        self.has_font = self.font_file.exists()
        
        self.confirmed_clips = {}
        self.device, self.model = self._initialize_model()
        self.compilation_log = []

    def _initialize_model(self):
        # This function is unchanged
        print("1. Initializing Whisper Model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda": print("✅ CUDA is available! Using GPU for transcription.")
        else: print("⚠️ CUDA not found. Using CPU. This will still be fast due to small segment processing.")
        print(f"   Loading Whisper model '{self.WHISPER_MODEL_SIZE}' onto {device.upper()}...")
        try:
            model = whisper.load_model(self.WHISPER_MODEL_SIZE, device=device)
            print("   Model loaded successfully.")
            if self.has_font: print(f"✅ Font found: {self.font_file}")
            else: print(f"⚠️ Font not found at 'fonts/CoolveticaRg.otf'. Text overlay will be disabled.")
            if self.ding_file.exists():
                if self.has_ding: print(f"✅ Ding sound enabled: {self.ding_file}")
                else: print("ℹ️ Ding sound found, but disabled by --no-ding flag.")
            else: print("⚠️ Ding sound not found at 'audio/ding.mp3'. Sound effects disabled.")
            return device, model
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            return None, None

    def run(self):
        # This function is unchanged
        if not self.model: return
        print(f"\n2. Searching for phrases in SRT files: {', '.join(f'\'{p}\'' for p in self.search_phrases)}")
        self._prepare_directories()
        coarse_matches = self._find_srt_matches()
        if not coarse_matches:
            print(f"\n❌ No mentions found in any SRT files in '{self.videos_dir}'.")
            return
        print(f"\n3. Found {len(coarse_matches)} potential mentions. Now refining with Whisper...")
        all_found_clips = []
        clip_counter = 0
        for i, match in enumerate(coarse_matches):
            print(f"\n--- Refining mention {i+1}/{len(coarse_matches)} from {match['video_file'].name} ---")
            word_start_time, word_end_time, context = self._refine_match_with_whisper(match)
            if word_start_time is not None:
                if self._is_duplicate(match['video_file'], word_start_time):
                    print(f"  ℹ️ Skipping duplicate mention confirmed at {word_start_time:.2f}s.")
                    continue
                clip_counter += 1
                padding_before = random.uniform(*self.CLIP_PADDING_BEFORE_S)
                padding_after = random.uniform(*self.CLIP_PADDING_AFTER_S)
                clip_start = max(0, word_start_time - padding_before)
                clip_end = word_end_time + padding_after
                clip_info = {
                    "id": clip_counter, "video_file": match['video_file'],
                    "phrase": match['phrase'], "start": clip_start, "end": clip_end,
                    "word_start_time": word_start_time, "word_end_time": word_end_time,
                    "padding_before": padding_before, "context": context, "srt_text": match['srt_text'],
                }
                output_path = self.temp_path / f"clip_{clip_counter:04d}.mp4"
                success = self._extract_clip(clip_info, output_path)
                if success:
                    all_found_clips.append(output_path)
                    self.compilation_log.append(clip_info)
                    if self.test_mode: self._save_test_clip(output_path, clip_info)
            else:
                print(f"  ❌ Could not confirm phrase '{match['phrase']}' in the audio segment.")
        if not all_found_clips:
            print("\n❌ No clips could be extracted after Whisper verification.")
            self.temp_dir.cleanup()
            return
        print(f"\n4. Creating final compilation video from {len(all_found_clips)} clips...")
        output_file = self._generate_output_filename()
        self._concatenate_clips(all_found_clips, output_file)
        self._save_compilation_log(output_file)
        self.temp_dir.cleanup()
        print("\n✨ Process complete. Temporary files have been removed.")
        
    def _extract_clip(self, clip_info, output_path):
        """
        Extracts a clip using a three-stage, timed text overlay for a "pulse" effect.
        """
        try:
            video_input = ffmpeg.input(str(clip_info['video_file']), ss=clip_info['start'], to=clip_info['end'])
            video_stream, audio_stream = video_input['v'], video_input['a']

            if self.has_font:
                # <-- NEW: This is the three-stage animation logic -->

                # Define the text strings for each state
                text_before_flash = f"'{clip_info['phrase'].replace(':', r'\\:')}' Counter"
                text_during_flash = f"{text_before_flash}: {clip_info['id']}"

                # Define the time windows for each state, relative to the clip's start time
                flash_start_time = max(0, clip_info['word_end_time'] - clip_info['start'])
                flash_end_time = flash_start_time + self.FLASH_DURATION_S

                # Common drawtext settings
                drawtext_settings = {
                    'fontfile': str(self.font_file),
                    'fontsize': 48,
                    'x': 'w-tw-20',
                    'y': 20,
                    'shadowx': 2,
                    'shadowy': 2
                }

                # 1. Pre-Flash State: Semi-transparent, no number
                video_stream = ffmpeg.drawtext(
                    video_stream,
                    text=text_before_flash,
                    fontcolor='white@0.5',
                    shadowcolor='black@0.5',
                    enable=f'lt(t,{flash_start_time})', # lt = less than
                    **drawtext_settings
                )
                
                # 2. Flash State: Fully opaque, with number
                video_stream = ffmpeg.drawtext(
                    video_stream,
                    text=text_during_flash,
                    fontcolor='white@1.0',
                    shadowcolor='black@1.0',
                    enable=f'between(t,{flash_start_time},{flash_end_time})',
                    **drawtext_settings
                )

                # 3. Post-Flash State: Semi-transparent, with number
                video_stream = ffmpeg.drawtext(
                    video_stream,
                    text=text_during_flash,
                    fontcolor='white@0.5',
                    shadowcolor='black@0.5',
                    enable=f'gte(t,{flash_end_time})', # gte = greater than or equal to
                    **drawtext_settings
                )
            
            if self.has_ding:
                ding_input = ffmpeg.input(str(self.ding_file))
                ding_delay_ms = max(0, (clip_info['padding_before'] - self.DING_OFFSET_S) * 1000)
                delayed_ding = ding_input.filter('adelay', f'{ding_delay_ms}|{ding_delay_ms}')
                audio_stream = ffmpeg.filter([audio_stream, delayed_ding], 'amix', duration='first')

            # Run ffmpeg with the final stream objects
            output_streams = (video_stream, audio_stream)
            (ffmpeg.output(*output_streams, str(output_path), vcodec='libx264', preset='fast', crf=23, acodec='aac', ar='44100')
             .run(capture_stdout=True, capture_stderr=True, quiet=self.FFMPEG_QUIET_MODE, overwrite_output=True))
             
            print(f"  ✅ Extracted final clip #{clip_info['id']} for '{clip_info['phrase']}'")
            return True
        except ffmpeg.Error as e:
            print(f"  ❌ FFMPEG Error extracting final clip #{clip_info['id']}:\n{e.stderr.decode()}")
            return False
            
    # --- All other functions are unchanged from the previous version ---
    def _refine_match_with_whisper(self, match):
        start_time = max(0, match['start_window'] - self.SRT_WINDOW_PADDING)
        end_time = match['end_window'] + self.SRT_WINDOW_PADDING
        temp_audio_path = self.temp_path / "segment.wav"
        try:
            ffmpeg.input(str(match['video_file']), ss=start_time, to=end_time).output(
                str(temp_audio_path), acodec='pcm_s16le', ac=1, ar='16000'
            ).run(capture_stdout=True, capture_stderr=True, quiet=self.FFMPEG_QUIET_MODE, overwrite_output=True)
        except ffmpeg.Error as e:
            print(f"  ❌ FFMPEG Error extracting audio segment:\n{e.stderr.decode()}")
            return None, None, None
        print(f"  Running Whisper on segment...")
        result = self.model.transcribe(str(temp_audio_path), word_timestamps=True, fp16=self.device=='cuda')
        all_words = self._get_word_list(result)
        temp_audio_path.unlink()
        if not all_words: return None, None, None
        num_all_words, phrase_words = len(all_words), match['phrase'].split()
        num_phrase_words = len(phrase_words)
        for i in range(num_all_words - num_phrase_words + 1):
            window = all_words[i : i + num_phrase_words]
            window_text = " ".join([self._normalize_text(w['word'].strip()) for w in window])
            if window_text == match['phrase']:
                word_start_time = start_time + window[0]['start']
                word_end_time = start_time + window[-1]['end']
                context = " ".join([w['word'].strip() for w in window])
                print(f"  ✅ Confirmed phrase at {word_start_time:.2f}s")
                return word_start_time, word_end_time, context
        return None, None, None

    def _is_duplicate(self, video_file, new_start_time):
        video_path_str = str(video_file)
        if video_path_str not in self.confirmed_clips:
            self.confirmed_clips[video_path_str] = []
        for existing_start_time in self.confirmed_clips[video_path_str]:
            if abs(new_start_time - existing_start_time) < self.DEDUPLICATION_SECONDS:
                return True
        self.confirmed_clips[video_path_str].append(new_start_time)
        return False
        
    def _find_srt_matches(self):
        coarse_matches = []
        srt_files = list(self.videos_dir.glob('*.srt'))
        for srt_file in srt_files:
            video_file = self._find_video_for_srt(srt_file)
            if not video_file: continue
            try:
                with open(srt_file, 'r', encoding='utf-8') as f: content = f.read()
                blocks = re.split(r'\n\s*\n', content.strip())
                for block in blocks:
                    lines = block.strip().split('\n')
                    if len(lines) < 3 or '-->' not in lines[1]: continue
                    timestamp_line, subtitle_text = lines[1], ' '.join(lines[2:]).strip()
                    for phrase in self.search_phrases:
                        if phrase in self._normalize_text(subtitle_text):
                            start_str, end_str = timestamp_line.split(' --> ')
                            coarse_matches.append({
                                "video_file": video_file, "phrase": phrase,
                                "start_window": self._parse_srt_timestamp(start_str),
                                "end_window": self._parse_srt_timestamp(end_str),
                                "srt_text": subtitle_text
                            })
            except Exception as e: print(f"  - Error reading {srt_file.name}: {e}")
        return coarse_matches

    def _find_video_for_srt(self, srt_path):
        base_name = srt_path.name.rsplit('.', 2)[0] if srt_path.name.endswith('.en.srt') else srt_path.stem
        for ext in self.VIDEO_EXTENSIONS:
            potential_video = srt_path.parent / (base_name + ext)
            if potential_video.exists(): return potential_video
        return None

    def _parse_srt_timestamp(self, ts_str):
        time_part, ms_part = ts_str.strip().split(',')
        h, m, s = map(int, time_part.split(':'))
        return h * 3600 + m * 60 + s + int(ms_part) / 1000

    def _normalize_text(self, text):
        return re.sub(r"[^\w\s]", "", text).lower().strip()

    def _prepare_directories(self):
        self.output_dir.mkdir(exist_ok=True)
        if self.test_mode:
            shutil.rmtree(self.clips_dir, ignore_errors=True)
            self.clips_dir.mkdir(parents=True)
            print(f"   🧪 TEST MODE: Individual clips will be saved to: {self.clips_dir.resolve()}")

    def _get_word_list(self, result):
        words = []
        for segment in result.get('segments', []):
            for word in segment.get('words', []):
                words.append({'word': word['word'], 'start': word['start'], 'end': word['end']})
        return words

    def _concatenate_clips(self, clip_files, output_file):
        concat_list_path = self.temp_path / 'concat.txt'
        with open(concat_list_path, 'w', encoding='utf-8') as f:
            for clip_path in sorted(clip_files):
                f.write(f"file '{clip_path.resolve()}'\n")
        try:
            ffmpeg.input(str(concat_list_path), format='concat', safe=0).output(
                str(output_file), c='copy'
            ).run(capture_stdout=True, capture_stderr=True, quiet=self.FFMPEG_QUIET_MODE, overwrite_output=True)
            print(f"  ✅ Compilation video saved as: {output_file.resolve()}")
        except ffmpeg.Error:
            print(f"  ⚠️ Concatenation with '-c copy' failed, retrying with re-encoding...")
            try:
                ffmpeg.input(str(concat_list_path), format='concat', safe=0).output(
                    str(output_file), vcodec='libx264', acodec='aac'
                ).run(overwrite_output=True, quiet=self.FFMPEG_QUIET_MODE)
                print(f"  ✅ Re-encoding successful. Compilation video saved as: {output_file.resolve()}")
            except ffmpeg.Error as e2:
                print(f"  ❌ FFMPEG Error on re-encoding as well:\n{e2.stderr.decode()}")
    
    def _save_test_clip(self, temp_clip_path, clip_info):
        safe_phrase = clip_info['phrase'].replace(' ', '_')
        test_clip_name = f"clip_{clip_info['id']:04d}_{safe_phrase}.mp4"
        test_clip_path = self.clips_dir / test_clip_name
        shutil.copy2(temp_clip_path, test_clip_path)
        metadata_file = test_clip_path.with_suffix('.json')
        serializable_info = clip_info.copy()
        serializable_info['video_file'] = str(serializable_info['video_file'])
        with open(metadata_file, 'w', encoding='utf-8') as f: json.dump(serializable_info, f, indent=4)

    def _generate_output_filename(self):
        safe_phrases = '_'.join(re.sub(r'[^\w-]', '', p) for p in self.search_phrases[:3])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.output_dir / f"compilation_{safe_phrases}_{timestamp}.mp4"

    def _save_compilation_log(self, output_file):
        log_filename = output_file.with_suffix('.json')
        summary = {
            "compilation_file": str(output_file.resolve()), "generation_time": datetime.now().isoformat(),
            "whisper_model": self.WHISPER_MODEL_SIZE, "device_used": self.device,
            "summary": {"total_clips_created": len(self.compilation_log), "phrases_searched": self.search_phrases},
            "clips": self.compilation_log
        }
        for clip in summary['clips']: clip['video_file'] = str(clip['video_file'])
        with open(log_filename, 'w', encoding='utf-8') as f: json.dump(summary, f, indent=4)
        print(f"  📝 Detailed compilation log saved as: {log_filename.resolve()}")

def main():
    parser = argparse.ArgumentParser(description='Whisper Mention Extractor - v13 (Pulse Animation)')
    parser.add_argument('--test', action='store_true', help='Test mode: Save individual clips to output/clips/ folder.')
    parser.add_argument('--no-ding', action='store_true', help="Disable the 'ding' sound effect.")
    parser.add_argument('phrases', nargs='*', help='Phrases to search for (e.g., "hello world").')
    args = parser.parse_args()

    if args.phrases: search_phrases = args.phrases
    else:
        print("Enter phrases to search for (one per line, press Enter twice when done):")
        search_phrases = []
        while True:
            try:
                phrase = input("  Phrase: ").strip()
                if not phrase: break
                search_phrases.append(phrase)
            except (EOFError, KeyboardInterrupt): break
    
    if not search_phrases:
        print("No phrases provided. Exiting.")
        return

    try:
        extractor = WhisperSrtHybridExtractor(
            search_phrases, test_mode=args.test, no_ding=args.no_ding
        )
        extractor.run()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()