#!/usr/bin/env python3
"""
Whisper Mention Extractor - v24 (Zoompan Syntax Fix)

- Fixes a crash in the outro animation by removing extra quotes from the 'zoompan'
  filter expression, allowing ffmpeg to correctly parse the animation math.
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
    # --- Main Configuration ---
    VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.webm'}
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg'}
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
    WHISPER_MODEL_SIZE = "base"
    
    # --- Timing & Animation Configuration ---
    SRT_WINDOW_PADDING = 2.0
    CLIP_PADDING_BEFORE_S = (0.4, 1.0)
    CLIP_PADDING_AFTER_S  = (0.7, 1.5)
    DEDUPLICATION_SECONDS = 2.0
    FLASH_DURATION_S = 0.3
    DING_OFFSET_S = 0.1
    BACKGROUND_MUSIC_VOLUME = 0.25
    FFMPEG_QUIET_MODE = True

    def __init__(self, search_phrases, test_mode=False, no_ding=False, music_dir=None):
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
        self.has_font = self.font_file.exists()
        
        self.music_dir = Path(music_dir) if music_dir else None
        self._check_assets()

        self.confirmed_clips = {}
        self.device, self.model = self._initialize_model()
        self.compilation_log = []

    def _check_assets(self):
        # This function is unchanged
        self.has_ding = self.ding_file.exists() and not self.no_ding
        self.has_bg_music = False
        self.has_outro = False
        if self.music_dir and self.music_dir.is_dir():
            if any(f.suffix.lower() in self.AUDIO_EXTENSIONS for f in self.music_dir.iterdir()):
                self.has_bg_music = True
        elif self.music_dir:
             print(f"⚠️ Music directory not found at '{self.music_dir}'.")
        outro_path = Path('outro').resolve()
        self.outro_music_path = outro_path / 'music'
        self.outro_bg_path = outro_path / 'backgrounds'
        self.outro_subscribe_path = outro_path / 'images' / 'subscribe_buttons'
        outro_music_ok = self.outro_music_path.is_dir() and any(self.outro_music_path.iterdir())
        outro_bg_ok = self.outro_bg_path.is_dir() and any(self.outro_bg_path.iterdir())
        outro_sub_ok = self.outro_subscribe_path.is_dir() and any(self.outro_subscribe_path.iterdir())
        if outro_path.is_dir() and outro_music_ok and outro_bg_ok and outro_sub_ok:
            self.has_outro = True
        else:
            print("ℹ️ Outro assets not found or incomplete. Outro will be skipped.")

    def _initialize_model(self):
        # This function is unchanged
        print("1. Initializing Whisper Model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda": print("✅ CUDA is available! Using GPU for transcription.")
        else: print("⚠️ CUDA not found. Using CPU.")
        print(f"   Loading Whisper model '{self.WHISPER_MODEL_SIZE}' onto {device.upper()}...")
        try:
            model = whisper.load_model(self.WHISPER_MODEL_SIZE, device=device)
            print("   Model loaded successfully.")
            if self.has_font: print(f"✅ Font found: {self.font_file}")
            else: print(f"⚠️ Font not found at 'fonts/CoolveticaRg.otf'.")
            if self.ding_file.exists():
                if self.has_ding: print(f"✅ Ding sound enabled: {self.ding_file}")
                else: print("ℹ️ Ding sound found, but disabled by --no-ding flag.")
            else: print("⚠️ Ding sound not found at 'audio/ding.mp3'.")
            if self.has_bg_music: print(f"✅ Background music directory found: {self.music_dir}")
            if self.has_outro: print("✅ Outro assets found. An outro will be generated.")
            return device, model
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            return None, None

    def run(self):
        # This function is unchanged
        if not self.model: return
        print(f"\n2. Searching for phrases in SRT files...")
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
            return
        print(f"\n4. Creating base compilation video...")
        base_compilation_path = self.temp_path / "base_compilation.mp4"
        self._concatenate_clips(all_found_clips, base_compilation_path)

        if not base_compilation_path.exists() or base_compilation_path.stat().st_size == 0:
            print("❌ Base compilation failed, no video was created. Aborting post-processing.")
            self.temp_dir.cleanup()
            return
            
        final_video_path = base_compilation_path
        try:
            probe = ffmpeg.probe(str(base_compilation_path))
            video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if not video_info:
                 print(f"❌ Could not get video stream from the base compilation. Skipping post-processing.")
                 shutil.copy2(final_video_path, self._generate_output_filename())
                 self.temp_dir.cleanup()
                 return
            width, height = video_info['width'], video_info['height']
            framerate_str = video_info.get('r_frame_rate', '30/1')
            num, den = map(int, framerate_str.split('/'))
            framerate = num / den
            duration = float(probe['format']['duration'])
        except (ffmpeg.Error, StopIteration, KeyError, ZeroDivisionError) as e:
            print(f"❌ Could not get properties of the base compilation. Skipping post-processing. Error: {e}")
            shutil.copy2(final_video_path, self._generate_output_filename())
            self.temp_dir.cleanup()
            return
        if self.has_bg_music:
            print("\n5. Processing background music...")
            self._check_and_wait_for_music(duration)
            music_compilation_path = self.temp_path / "with_music.mp4"
            if self._add_background_music(final_video_path, music_compilation_path):
                final_video_path = music_compilation_path
        if self.has_outro:
            print("\n6. Generating and attaching animated outro...")
            outro_clip_path = self.temp_path / "outro.mp4"
            if self._create_outro_clip(outro_clip_path, width, height, framerate):
                final_stitched_path = self.temp_path / "final_stitched.mp4"
                if self._stitch_outro(final_video_path, outro_clip_path, final_stitched_path):
                    final_video_path = final_stitched_path
        output_file = self._generate_output_filename()
        shutil.copy2(final_video_path, output_file)
        print(f"\n✨ Process complete! Final video saved as: {output_file}")
        self._save_compilation_log(output_file)
        self.temp_dir.cleanup()

    def _create_outro_clip(self, output_path, width, height, framerate):
        """Generates the outro with a 'zoompan' filter for the scaling animation."""
        try:
            outro_music = random.choice([f for f in self.outro_music_path.iterdir()])
            outro_bg = random.choice([f for f in self.outro_bg_path.iterdir()])
            subscribe_button = random.choice([f for f in self.outro_subscribe_path.iterdir()])
            print(f"  - Building {width}x{height} @ {framerate:.2f}fps outro...")
            outro_duration = float(ffmpeg.probe(str(outro_music))['format']['duration'])
            
            bg_input = (
                ffmpeg.input(str(outro_bg), loop=1, t=outro_duration, framerate=framerate)
                .filter('scale', width, height, force_original_aspect_ratio='decrease')
                .filter('pad', width, height, '(ow-iw)/2', '(oh-ih)/2')
                .filter('setsar', 1)
            )

            button_input = ffmpeg.input(str(subscribe_button))
            music_input = ffmpeg.input(str(outro_music))

            anchor_x = width * random.uniform(0.4, 0.6)
            anchor_y = height * random.uniform(0.4, 0.6)

            # <-- MODIFIED: Removed extra quotes from the zoom expression -->
            zoom_expr = "1.05+0.25*sin(in_time*3)" 

            animated_button = (
                button_input
                .filter('scale', 450, 225, force_original_aspect_ratio='decrease')
                .filter('zoompan', z=zoom_expr, x='iw/2-(iw/zoom/2)', y='ih/2-(ih/zoom/2)', d=99999, s=f'{width}x{height}', fps=framerate)
                .filter('rotate', "20*(PI/180)*sin(t*2.5)", fillcolor='none', ow='rotw(a)', oh='roth(a)')
            )

            video_stream = bg_input.overlay(
                animated_button,
                x=f'{anchor_x} - W/2',
                y=f'{anchor_y} - H/2',
                eof_action='pass'
            )
            
            video_stream = video_stream.filter('fade', type='out', start_time=outro_duration - 5, duration=5)
            audio_stream = music_input['a'].filter('afade', type='out', start_time=outro_duration - 5, duration=5)
            
            (ffmpeg.output(video_stream, audio_stream, str(output_path), vcodec='libx264', acodec='aac', pix_fmt='yuv420p', r=framerate)
             .run(quiet=self.FFMPEG_QUIET_MODE, overwrite_output=True))
            return True
        except ffmpeg.Error as e:
            print(f"  ❌ Error creating outro clip. FFmpeg stderr:\n{e.stderr.decode()}")
            return False

    # --- All other functions are unchanged from here down ---
    def _stitch_outro(self, main_clip_path, outro_clip_path, output_path):
        try:
            main_clip = ffmpeg.input(str(main_clip_path))
            outro_clip = ffmpeg.input(str(outro_clip_path))
            main_duration = float(ffmpeg.probe(str(main_clip_path))['format']['duration'])
            v1 = main_clip['v'].filter('setsar', 1)
            a1 = main_clip['a'].filter('aformat', channel_layouts='stereo', sample_rates='44100')
            v2 = outro_clip['v'].filter('setsar', 1)
            a2 = outro_clip['a'].filter('aformat', channel_layouts='stereo', sample_rates='44100')
            video_stream = ffmpeg.filter([v1, v2], 'xfade', transition='fade', duration=1, offset=main_duration - 1)
            audio_stream = ffmpeg.filter([a1, a2], 'acrossfade', duration=1)
            (ffmpeg.output(video_stream, audio_stream, str(output_path), vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
             .run(quiet=self.FFMPEG_QUIET_MODE, overwrite_output=True))
            return True
        except ffmpeg.Error as e:
            print(f"  ❌ Error stitching outro. FFmpeg stderr:\n{e.stderr.decode()}")
            return False

    def _add_background_music(self, input_video_path, output_video_path):
        if not self.music_dir:
            return False
        try:
            music_files = [f for f in self.music_dir.iterdir() if f.is_file() and f.suffix.lower() in self.AUDIO_EXTENSIONS]
            if not music_files:
                print("  - No music files found in the specified directory.")
                return False
            random.shuffle(music_files)
            print("  - Shuffling and concatenating background music tracks...")
            concat_list_path = self.temp_path / 'music_concat.txt'
            with open(concat_list_path, 'w', encoding='utf-8') as f:
                for music_file in music_files:
                    f.write(f"file '{music_file.resolve()}'\n")
            full_music_track_path = self.temp_path / "full_bg_music.mp3"
            (ffmpeg.input(str(concat_list_path), format='concat', safe=0)
             .output(str(full_music_track_path), acodec='libmp3lame')
             .run(quiet=self.FFMPEG_QUIET_MODE, overwrite_output=True))
            print("  - Mixing music with compilation audio...")
            main_video = ffmpeg.input(str(input_video_path))
            music_track = ffmpeg.input(str(full_music_track_path))
            music_audio = music_track['a'].filter('volume', self.BACKGROUND_MUSIC_VOLUME)
            mixed_audio = ffmpeg.filter(
                [main_video['a'], music_audio], 'amix',
                inputs=2, duration='longest'
            )
            (ffmpeg.output(main_video['v'], mixed_audio, str(output_video_path), vcodec='copy', acodec='aac', shortest=None)
             .run(quiet=self.FFMPEG_QUIET_MODE, overwrite_output=True))
            return True
        except ffmpeg.Error as e:
            print(f"  ❌ Error adding background music. FFmpeg stderr:\n{e.stderr.decode()}")
            return False

    def _check_and_wait_for_music(self, video_duration):
        if not self.music_dir:
            return
        while True:
            music_files = [f for f in self.music_dir.iterdir() if f.is_file() and f.suffix.lower() in self.AUDIO_EXTENSIONS]
            if not music_files: total_music_duration = 0
            else:
                print("  - Calculating total music duration...")
                total_music_duration = sum(float(ffmpeg.probe(str(f))['format']['duration']) for f in music_files)
            if total_music_duration >= video_duration:
                print(f"  - ✅ Sufficient music found: {total_music_duration:.2f}s available for a {video_duration:.2f}s video.")
                break
            else:
                shortfall = video_duration - total_music_duration
                print(f"\n🔴 Music shortfall detected! Please add at least {shortfall:.2f} more seconds of music to:\n   '{self.music_dir.resolve()}'")
                input("   Press Enter to check again...")

    def _extract_clip(self, clip_info, output_path):
        try:
            video_input = ffmpeg.input(str(clip_info['video_file']), ss=clip_info['start'], to=clip_info['end'])
            video_stream, audio_stream = video_input['v'], video_input['a']
            if self.has_font:
                text_before_flash = f"'{clip_info['phrase'].replace(':', r'\\:')}' Counter"
                text_during_flash = f"{text_before_flash}: {clip_info['id']}"
                flash_start_time = max(0, clip_info['word_end_time'] - clip_info['start'])
                flash_end_time = flash_start_time + self.FLASH_DURATION_S
                drawtext_settings = {
                    'fontfile': str(self.font_file), 'fontsize': 48,
                    'x': 'w-tw-20', 'y': 20, 'shadowx': 2, 'shadowy': 2
                }
                video_stream = ffmpeg.drawtext(
                    video_stream, text=text_before_flash, fontcolor='white@0.5',
                    shadowcolor='black@0.5', enable=f'lt(t,{flash_start_time})', **drawtext_settings
                )
                video_stream = ffmpeg.drawtext(
                    video_stream, text=text_during_flash, fontcolor='white@1.0',
                    shadowcolor='black@1.0', enable=f'between(t,{flash_start_time},{flash_end_time})', **drawtext_settings
                )
                video_stream = ffmpeg.drawtext(
                    video_stream, text=text_during_flash, fontcolor='white@0.5',
                    shadowcolor='black@0.5', enable=f'gte(t,{flash_end_time})', **drawtext_settings
                )
            if self.has_ding:
                ding_input = ffmpeg.input(str(self.ding_file))
                ding_delay_ms = max(0, (clip_info['padding_before'] - self.DING_OFFSET_S) * 1000)
                delayed_ding = ding_input.filter('adelay', f'{ding_delay_ms}|{ding_delay_ms}')
                audio_stream = ffmpeg.filter([audio_stream, delayed_ding], 'amix', duration='first')
            (ffmpeg.output(video_stream, audio_stream, str(output_path), vcodec='libx264', preset='fast', crf=23, acodec='aac', ar='44100')
             .run(capture_stdout=True, capture_stderr=True, quiet=self.FFMPEG_QUIET_MODE, overwrite_output=True))
            print(f"  ✅ Extracted final clip #{clip_info['id']} for '{clip_info['phrase']}'")
            return True
        except ffmpeg.Error as e:
            print(f"  ❌ FFMPEG Error extracting final clip #{clip_info['id']}:\n{e.stderr.decode()}")
            return False

    def _refine_match_with_whisper(self, match):
        start_time = max(0, match['start_window'] - self.SRT_WINDOW_PADDING)
        end_time = match['end_window'] + self.SRT_WINDOW_PADDING
        temp_audio_path = self.temp_path / "segment.wav"
        try:
            ffmpeg.input(str(match['video_file']), ss=start_time, to=end_time).output(
                str(temp_audio_path), acodec='pcm_s16le', ac=1, ar='16000'
            ).run(capture_stdout=True, capture_stderr=True, quiet=self.FFMPEG_QUIET_MODE, overwrite_output=True)
        except ffmpeg.Error as e:
            return None, None, None
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
        except ffmpeg.Error:
            try:
                ffmpeg.input(str(concat_list_path), format='concat', safe=0).output(
                    str(output_file), vcodec='libx264', acodec='aac'
                ).run(overwrite_output=True, quiet=self.FFMPEG_QUIET_MODE)
            except ffmpeg.Error as e2:
                print(f"  ❌ FFMPEG Error concatenating clips:\n{e2.stderr.decode()}")
    
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
    parser = argparse.ArgumentParser(description='Whisper Mention Extractor - v23 (Zoompan Animation)')
    parser.add_argument('--test', action='store_true', help='Test mode: Save individual clips to output/clips/ folder.')
    parser.add_argument('--no-ding', action='store_true', help="Disable the 'ding' sound effect.")
    parser.add_argument('--music', type=str, help="Path to a directory containing background music files.")
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
            search_phrases,
            test_mode=args.test,
            no_ding=args.no_ding,
            music_dir=args.music
        )
        extractor.run()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()