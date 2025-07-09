# Mention Extractor

A Python tool that searches for specific words or phrases in video subtitle files (.srt) and creates a compilation video of all mentions using ffmpeg with **precise audio-based timing**.

## Features

### 🔥 **NEW: Precision Audio Analysis**
- **Multiple Phrase Search**: Enter multiple phrases (e.g., "Trump", "President Trump")
- **TTS-Based Timing**: Uses local text-to-speech to generate reference audio
- **Audio Cross-Correlation**: Finds exact timing of phrases within subtitle segments
- **Quarter-Second Precision**: Clips are trimmed to 0.25s before + phrase + 0.25s after
- **Smart Counter**: Accurately counts multiple mentions within single subtitle fragments

### 🎬 **Core Features**
- Searches through all .srt subtitle files in `videos/` folder
- Finds every mention of user-specified phrases (case-insensitive)
- Extracts ultra-precise video clips using audio analysis
- Adds visual counter overlay with drop shadow (e.g., "Trump Counter: 15")
- Plays ding sound effect before each phrase (requires `audio/ding.mp3`)
- Creates tight compilation videos with no wasted time
- Supports multiple video formats (.mp4, .avi, .mkv, .mov, .wmv)

## Prerequisites

- Python 3.8 or higher
- ffmpeg installed and available in PATH
- Required Python packages (see Installation)

## Installation

1. **Install Python dependencies:**
   ```bash
   # On Windows, run:
   install_requirements.bat
   
   # Or manually:
   pip install -r requirements.txt
   ```

2. **Install ffmpeg:**

**Windows:**
1. Download ffmpeg from https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

## Usage

1. **Setup folder structure:**
   - Create a `videos/` folder
   - Place your video files (.mp4, .avi, etc.) and subtitle files (.srt) in the `videos/` folder
   - Optionally add `audio/ding.mp3` for sound effects

2. **Run the program:**
   ```bash
   python mention_extractor.py
   ```

3. **Enter your search phrases:**
   - Input multiple phrases (one per line)
   - Press Enter twice when done
   - Example: "Trump", "President Trump", "Donald"

4. **Watch the magic happen:**
   - TTS audio generation for phrase matching
   - Audio analysis to find exact phrase timing
   - Ultra-precise clip extraction (0.25s padding)
   - Compilation video creation: `compilation_[phrases]_[timestamp].mp4`

## Example

```
=== Mention Extractor ===
This tool will search for words/phrases in video subtitles
and create a compilation video of all mentions.

Enter phrases to search for (one per line, press Enter twice when done):
  Phrase: Trump
  Phrase: President Trump  
  Phrase: 

Searching for 2 phrases: 'Trump', 'President Trump'

Found 6 subtitle files in videos/ folder:
  - videos/"The next move will change history" [dUjoAQoMKmY].en.srt
  - videos/"Promises made, promises kept" [ukql8W5IA00].en.srt
  ...

Generating TTS audio for phrase matching...
  ✅ Generated TTS for: 'Trump'
  ✅ Generated TTS for: 'President Trump'

Found 25 subtitle fragments containing 'Trump', 'President Trump'
After removing overlaps: 20 clips containing 37 total mentions

Creating compilation video with precise phrase timing...

Processing clip 1: And Trump said that America needs...
  Analyzing audio for phrase: 'Trump'
  ✅ Found 'Trump' at 12.34s (correlation: 0.847)
  ✅ Using audio match for 'Trump' (correlation: 0.847)
  Creating precise clip: 12.09s to 13.21s (duration: 1.12s)
  Counter: now at 1

Processing clip 2: President Trump made the decision...
  Analyzing audio for phrase: 'President Trump'
  ✅ Found 'President Trump' at 8.76s (correlation: 0.923)
  ✅ Using audio match for 'President Trump' (correlation: 0.923)
  Creating precise clip: 8.51s to 10.18s (duration: 1.67s)
  Counter: now at 2

Done! Your compilation video 'compilation_Trump_President_Trump_20241212_143055.mp4' 
contains 20 clips with 37 total mentions of 'Trump', 'President Trump'
```

## File Structure

The program expects your files to be organized like this:
```
your_directory/
├── mention_extractor.py
├── requirements.txt
├── install_requirements.bat  # Windows installer
├── audio/
│   └── ding.mp3             # Optional: sound effect for each mention
├── fonts/
│   └── CoolveticaRg.otf     # Font for counter overlay
└── videos/                  # All videos and subtitles go here
    ├── video1.mp4
    ├── video1.en.srt
    ├── video2.mp4
    ├── video2.en.srt
    └── ...
```

The subtitle files should have the same base name as their corresponding video files. The `audio/ding.mp3` file is optional but recommended for the full experience.

## Technical Details

### 🔬 **Audio Analysis Pipeline**
- **TTS Generation**: Uses `pyttsx3` to generate reference audio for each search phrase
- **Audio Extraction**: FFmpeg extracts 16kHz mono audio from video segments  
- **Cross-Correlation**: `scipy.signal.correlate` finds exact phrase timing within segments
- **Precision Timing**: Clips trimmed to 0.25s before phrase + phrase duration + 0.25s after

### 🎬 **Video Processing**
- **Search**: Case-insensitive substring matching with multiple phrase support
- **Overlap Removal**: Smart merging of overlapping subtitle segments
- **Counter Logic**: Accurate counting of multiple mentions within single fragments
- **Visual Overlay**: White text with drop shadow in top-right corner (e.g., "Trump Counter: 15")
- **Audio Mixing**: Ding sound mixed with original audio before each phrase
- **Fallback Mode**: If TTS fails, falls back to subtitle-based timing

### ⚙️ **Output Specifications**
- **Format**: MP4 with H.264/AAC encoding
- **Video**: 30fps, original resolution, CRF 23 quality
- **Audio**: 48kHz stereo, perfect sync with video
- **Cleanup**: All temporary files automatically deleted after processing

## Troubleshooting

- **"ffmpeg is not installed"**: Make sure ffmpeg is installed and in your system PATH
- **"No video file found"**: Ensure video files have the same base name as .srt files
- **"No SRT files found"**: Make sure subtitle files are in the same directory as the script
- **Extraction errors**: Check that video files aren't corrupted and timestamps are valid

## Notes

- The program re-encodes all clips to ensure perfect audio/video synchronization and prevent corruption
- Processing time is longer due to re-encoding, but results in much higher quality output
- All clips are standardized to 30fps and 48kHz stereo audio for consistency
- Large numbers of matches will take longer to process due to re-encoding
- The output filename includes a timestamp to avoid overwriting previous compilations 