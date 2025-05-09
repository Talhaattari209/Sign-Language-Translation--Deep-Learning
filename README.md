---
title: Sign Language Translator
emoji: 🤟
colorFrom: indigo
colorTo: indigo
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
---

# Sign Language Translator

A powerful web application that enables translation between text and sign language videos. This project provides a user-friendly interface for translating text to sign language videos and vice versa, supporting multiple languages and sign language formats.

## Features

- Text to Sign Language Translation
- Support for multiple languages (Urdu, English)
- Video generation for sign language
- Interactive web interface
- Real-time translation capabilities
- Support for various sign language formats

## Requirements

- Python 3.7+
- FFMPEG (for video processing)
- Required Python packages:
  - streamlit
  - sign_language_translator
  - opencv-python
  - requests

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SL-PSL-RTP.git
cd SL-PSL-RTP
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install FFMPEG:
- Windows: Download from [FFMPEG website](https://ffmpeg.org/download.html)
- Linux: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Use the interface to:
   - Enter text for translation
   - Select source and target languages
   - Generate sign language videos
   - View translation results

## Technical Details

The application uses:
- Streamlit for the web interface
- Sign Language Translator library for core translation functionality
- FFMPEG for video processing
- Advanced language models for accurate translations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Sign Language Translator library
- Streamlit team
- All contributors and supporters
