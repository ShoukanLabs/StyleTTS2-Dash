import os
import re

import librosa
import soundfile as sf

import numpy as np
import torch
import torchaudio
import yaml

from tqdm import tqdm

from .diffusion.sampler import DiffusionSampler, KarrasSchedule, ADPM2Sampler
from .models.stts2 import build_model
from VoPho.engine import Phonemizer

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len(symbols)):
    dicts[symbols[i]] = i


class TextCleaner:
    def __init__(self, dummy=None):
        global dicts
        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            if char in self.word_index_dictionary.keys():
                indexes.append(self.word_index_dictionary[char])
        return indexes


def split_and_recombine_text(text, desired_length=200, max_length=300):
    """
    Split text into chunks of a desired length trying to keep sentences intact.
    Text wrapped in language tags <lang>...</lang> will not be split.
    """
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[""]', '"', text)

    # First, find all language tag blocks and replace them with placeholders
    tag_pattern = r'<([^>]+)>(.*?)</\1>'
    protected_blocks = []

    def replace_tag(match):
        protected_blocks.append(match.group(0))
        return f"__PROTECTED_BLOCK_{len(protected_blocks) - 1}__"

    processed_text = re.sub(tag_pattern, replace_tag, text, flags=re.DOTALL)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(processed_text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += processed_text[pos]
            if processed_text[pos] == '"':
                in_quote = not in_quote
        return processed_text[pos]

    def peek(delta):
        p = pos + delta
        return processed_text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # Check if we're at the start of a protected block
        if c == '_' and current.endswith('__PROTECTED_BLOCK_'):
            # Find the end of the placeholder
            while pos < end_pos and not processed_text[pos:pos + 2] == '__':
                c = seek(1)
            c = seek(1)  # Get past the last _
            continue

        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in '!?.\n ' and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in '!?\n' or (c == '.' and peek(1) in '\n ')):
            # seek forward if we have consecutive boundary markers but still within the max length
            while pos < len(processed_text) - 1 and len(current) < max_length and peek(1) in '!?.':
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in '\n ':
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r'^[\s\.,;:!?]*$', s)]

    # Restore protected blocks
    def restore_blocks(text):
        for i, block in enumerate(protected_blocks):
            text = text.replace(f"__PROTECTED_BLOCK_{i}__", block)
        return text

    rv = [restore_blocks(chunk) for chunk in rv]

    return rv


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


#### STTS2 PIPELINE ####

class StyleTTS2Pipeline:
    def __init__(self):
        """
        The StyleTTS2 pipeline, please us .load_from_folder to load the model before continuing...
        """
        self.is_vokanv2 = False
        self.model = None
        self.phonemizer = Phonemizer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = None
        self.sampler = None

    def load_from_files(self, path_to_model, path_to_config, is_vokanv2=False, is_tsukasa=False, map_location=None):
        """
        Loads the model located in the folder into the pipeline for usage

        :param path_to_model: The path to the model checkpoint
        :param path_to_config: The path to the model config
        :param is_vokanv2: Whether the model is a VokanV2 model or not
        :param is_tsukasa: Whether the model is a soshyant tsukasa model or not
        :param map_location: The device to load the model on
        """
        print("loading config...")
        config = yaml.safe_load(open(path_to_config))
        self.config = config
        model = build_model(config['model_params'], is_vokan=is_vokanv2, is_tsukasa=is_tsukasa)

        _ = [model[key].eval() for key in model]
        if map_location:
            self.device = "cpu"
            _ = [model[key].to(self.device) for key in model]
        else:
            _ = [model[key].to(self.device) for key in model]

        print("loading model...")
        for key, state_dict in torch.load(path_to_model,
                                          map_location=self.device,
                                          weights_only=True)['net'].items():
            if not key in model:
                continue
            try:
                model[key].load_state_dict(state_dict)
            except:
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                model[key].load_state_dict(state_dict, strict=False)

        self.model = model
        self.is_vokanv2 = is_vokanv2

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
            clamp=False
        )

        print("Done!")

    def preprocess(self, wave):
        """
        Turns audio into a Mel Spectrogram supported by the config

        :param wave: The numpy audio to preprocess before inference
        :return:
        """

        wave_tensor = torch.from_numpy(wave).float()
        to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=self.config["model_params"]["n_mels"],
            n_fft=self.config["preprocess_params"]["spect_params"]["n_fft"],
            win_length=self.config["preprocess_params"]["spect_params"]["win_length"],
            hop_length=self.config["preprocess_params"]["spect_params"]["hop_length"])
        mean, std = -4, 4
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor

    def compute_style(self, path):
        """
        Computes the style vector for the audio at the given path

        :param path: The path to the audio file
        :return:
        """
        wave, sr = librosa.load(path, sr=self.config["preprocess_params"]["sr"])
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != self.config["preprocess_params"]["sr"]:
            audio = librosa.resample(audio, sr, self.config["preprocess_params"]["sr"])
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    @torch.no_grad()
    def forward(self, tokens, ref_s, prev_s, alpha, beta, t, diffusion_steps, embedding_scale, speed):
        """
        The forward method, does all the actual TTS inference

        :param tokens: The list of tokens from the TextCleaner class
        :param ref_s: The reference style vector
        :param prev_s: The style vector from the previous generation (if longform)
        :param alpha: The alpha for the generation
        :param beta: The beta for the generation
        :param t: The ratio between the old reference and the new generated vector
        :param diffusion_steps: The amount of diffusion steps
        :param embedding_scale: The embedding scale, higher is unstable but more expressive
        :param speed: The amount to speed up or slow down the speech (1 = normal, 1.1 = 10% faster)
        :return: (audio, the style vector to use as prev_s)
        """

        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                  embedding=bert_dur,
                                  embedding_scale=embedding_scale,
                                  features=ref_s,  # reference from the same speaker as the embedding
                                  num_steps=diffusion_steps).squeeze(1)

            if prev_s is not None:
                # convex combination of previous and current style
                s_pred = t * prev_s + (1 - t) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            s_pred = torch.cat([ref, s], dim=-1)

            d = self.model.predictor.text_encoder(d_en,
                                                  s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            duration = duration * 1 / speed
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.config["model_params"]["decoder"]["type"] == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.config["model_params"]["decoder"]["type"] == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,
                                     F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy(), s_pred

    def postprocess(self, audio, threshold=95, max_samples=20000, lead_percent=0.08, trail_percent=0.08):
        """
        The post process method, cleans up any artefacts from generation

        :param audio: The numpy wave
        :param threshold: The silence threshold
        :param max_samples: The max amount of samples that can be cut
        :param lead_percent: The leading percent (in decimals) to fade out
        :param trail_percent: The trailing  percent (in decimals) to fade out
        :return: (sr, audio)
        """

        # S-Curve
        np_log_99 = np.log(99)

        def s_curve(p):
            assert 0 <= p and p <= 1, p
            if p == 0 or p == 1:
                return p
            p = (2 * p - 1) * np_log_99
            s = 1 / (1 + np.exp(-p))
            s = (s - 0.01) * 50 / 49
            assert 0 <= s and s <= 1, s
            return s

        # Post-Processing
        thresh = np.percentile(np.abs(audio), threshold)
        CUT_SAMPLES = max_samples  # max samples to cut, in practice only 4-6k are actually cut
        lead_percent = lead_percent
        trail_percent = trail_percent

        # Leading artefact removal
        left = CUT_SAMPLES + int(len(audio) * lead_percent)
        for j in range(left):
            if abs(audio[j]) > thresh:
                left = j
                break

        left = max(0, min(left - int(len(audio) * lead_percent), CUT_SAMPLES))
        audio[:left] = 0
        for k in range(int(len(audio) * lead_percent)):
            s = s_curve(k / int(len(audio) * lead_percent))
            audio[k + left] *= s

        # Trailing artefact removal
        right = len(audio) - CUT_SAMPLES - int(len(audio) * trail_percent)
        for j in range(len(audio) - 1, right, -1):
            if abs(audio[j]) > thresh:
                right = j
                break

        right = min(len(audio), max(right + int(len(audio) * trail_percent), len(audio) - CUT_SAMPLES))
        audio[right:] = 0
        for k in range(int(len(audio) * trail_percent)):
            s = s_curve(k / int(len(audio) * trail_percent))
            audio[right - int(len(audio) * trail_percent) + k] *= (1 - s)

        return self.config["preprocess_params"]["sr"], audio

    def generate(self,
                 text,
                 style,
                 long_form_identity_ratio=0.7,
                 alpha=0.3,
                 beta=0.3,
                 diffusion_steps=5,
                 embedding_scale=1,
                 speed=1,
                 language=None,
                 post_processing_args=None,
                 output_file_path=None):
        """
        :param text: The input text
        :param style: The path to an audio file, or a tensor for the style vector
        :param long_form_identity_ratio: The identity preservation ratio in longform generation
        :param alpha: The alpha for the generation
        :param beta: The beta for the generation
        :param diffusion_steps: The amount of diffusion steps
        :param embedding_scale: The embedding scale (higher = more expressive but more unstable)
        :param speed: The amount to speed up or slow down the speech (1 = normal, 1.1 = 10% faster)
        :param language: The language code (i.e en, zh, it) to force phonemize in, if None, will use multicode support
        :param post_processing_args: any post processing args (in a dict)
        :param output_file_path: the output file path, if any (will still return a numpy wav)
        :return: (sr, numpy audio)
        """

        if post_processing_args is None:
            post_processing_args = {}
        audio = np.array([])
        s_prev = None

        textcleaner = TextCleaner()

        if not isinstance(style, str):
            pass
        else:
            style = self.compute_style(style)

        texts = split_and_recombine_text(text, max_length=300)
        if len(texts) > 1:
            texts = tqdm(texts)
        for text in texts:
            if language:
                phonemes = self.phonemizer.phonemize_for_language(text, language)
            elif language == "phonemes":
                phonemes = text
            else:
                phonemes = self.phonemizer.phonemize(text)

            phonemes = textcleaner(phonemes)

            synthesized_audio, s_prev = self.forward(tokens=phonemes,
                                                     ref_s=style,
                                                     prev_s=s_prev,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     t=long_form_identity_ratio,
                                                     diffusion_steps=diffusion_steps,
                                                     embedding_scale=embedding_scale,
                                                     speed=speed)

            _, synthesized_audio = self.postprocess(synthesized_audio, **post_processing_args)

            audio = np.concatenate((audio, synthesized_audio))

        scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)

        if output_file_path:
            sf.write(output_file_path, scaled, self.config["preprocess_params"]["sr"])  # a wav at sampling rate 1 Hz

        return self.config["preprocess_params"]["sr"], scaled
