# MemorySystem

MemorySystem is a hardware-independent framework designed to manage user-specific long-term memory for social robots. It provides a modular architecture to handle audio processing, speaker identification, large language model (LLM) integration, and vector-based information retrieval.

## Background

This project was developed as part of a Bachelor's Thesis. It is no longer actively maintained and will not receive future updates. Users are free to use the system at their own discretion.

## Features

- **Hardware Independence:** Designed to run on various platforms supporting Python.
- **Long-term Memory:** Persistent storage and retrieval of user-specific information.
- **Audio Pipeline:** Integrated noise reduction, speaker identification, and speech-to-text.
- **Modular API:** Clean separation between interaction logic, models, and databases.

## Prerequisites

### System Dependencies (Linux)
Before installing the Python packages, ensure your system has the necessary audio headers and multimedia libraries:

```bash
# Install audio development headers for PyAudio
sudo apt update
sudo apt install libasound-dev portaudio19-dev

# Install FFmpeg for audio processing (required by pydub, librosa, and faster-whisper)
sudo apt install ffmpeg
```
## Installation Guide

```bash
git clone https://github.com/Tim-Mueller-Bagehl/MemorySystem.git
cd MemorySystem
pip install -e .
```

## Quick Start

Here's a basic usage example:

```python

from MemorySystem import InteractionManager

system = InteractionManager()

system.start()

```

If you want to customise your Interaction:

```python
from MemorySystem import InteractionManager

system = InteractionManager()

system.handleManualInput(ID,audiofile,transcript)
```

A new User has to register first:

```python

from MemorySystem import InteractionManager

system = InteractionManager()

audiofile = "Path to your File" #should contain Info about the User and be 60 Seconds long

ID = system.addNewPerson(audiofile)