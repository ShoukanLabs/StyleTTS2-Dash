from STTS2Dash.tts import StyleTTS2Pipeline
import torch

if __name__ == '__main__':
    tts = StyleTTS2Pipeline()
    tts.load_from_files("./test_models/epoch 7.pth",
                        "./test_models/config_n.yml", is_tsukasa=True)

    while True:
        text = input("Enter text to synthesize (or 'quit' to exit): ").strip()

        if text.lower() == 'quit':
            break

        if not text:
            print("Text cannot be empty. Please try again.")
            continue

        # Default parameters
        diffusion_steps = 30
        alpha = 0.2
        beta = 0.4
        embedding_scale = 2
        audio_style_path = "./test_models/audio.wav"

        # Parameter customization
        while True:
            customize = input("Customize generation parameters? (y/n): ").strip().lower()

            if customize == 'y':
                try:
                    alpha = float(input("Enter alpha (current: {0}): ".format(alpha)) or alpha)
                    beta = float(input("Enter beta (current: {0}): ".format(beta)) or beta)
                    embedding_scale = float(
                        input("Enter embedding scale (current: {0}): ".format(embedding_scale)) or embedding_scale)
                    diffusion_steps = int(
                        input("Enter diffusion steps (current: {0}): ".format(diffusion_steps)) or diffusion_steps)

                    # New option for audio style path
                    audio_style_path = input(
                        "Enter audio style path (current: {0}): ".format(audio_style_path)) or audio_style_path

                    break
                except ValueError:
                    print("Invalid input. Please enter numeric values for numeric parameters.")
                    continue
            elif customize == 'n':
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        # Generate audio
        try:
            tts.generate(text,
                         audio_style_path,
                         diffusion_steps=diffusion_steps,
                         alpha=alpha,
                         beta=beta,
                         embedding_scale=embedding_scale,
                         output_file_path="./test.wav",
                         language="en")
            print(f"Done... Saved audio with parameters:")
            print(f"Audio Style Path: {audio_style_path}")
            print(
                f"Alpha: {alpha}, Beta: {beta}, Embedding Scale: {embedding_scale}, Diffusion Steps: {diffusion_steps}")
        except Exception as e:
            print(f"Error generating audio: {e}")

    print("Text-to-speech generation ended.")
