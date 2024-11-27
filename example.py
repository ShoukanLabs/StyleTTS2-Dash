from STTS2Dash.tts import StyleTTS2Pipeline
import torch

if __name__ == '__main__':
    tts = StyleTTS2Pipeline()
    tts.load_from_files("./test_models/vokan.pth",
                        "./test_models/config.yml")

    tts.generate("This is a test for Style TTS dash. a tiny inference library for Style TTS 2 and vokan models",
                 "./test_models/Patrick Bateman.wav",
                 diffusion_steps=15,
                 alpha=0.3,
                 beta=0.6,
                 embedding_scale=2,
                 output_file_path="./test.wav",
                 language="en")
