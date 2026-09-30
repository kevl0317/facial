"""facial: annotate how a person speaks in a video.

MediaPipe measures hands, face and pose frame by frame; audio analysis measures
the voice; Claude fuses those measurements with the subtitles into a per-window
reading (confidence / focus / tension, intent, emotion arc) that is rendered
back onto the video.
"""

__version__ = "0.1.0"
