from setup import setup_video_capture
from keyboard_parser import KeyboardParser
from keyboard_parser2 import KeyboardParser2

# parser = KeyboardParser()
parser = KeyboardParser2()
setup_video_capture(parser.process, "videos/Machine Love - Jamie Paige (Piano Tutorial) [PO0gU5QVKFk].webm")