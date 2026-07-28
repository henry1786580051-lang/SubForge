"""WhisperX forced-alignment model registry.

Keep the language-to-model mapping independent from the optional WhisperX
runtime so the desktop backend can show and download models without importing
PyTorch during startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AlignmentModelSpec:
    id: str
    language: str
    language_name: str
    model_name: str
    source: str
    filename: str = ""
    url: str = ""
    size: str = ""


_TORCHAUDIO_BASE = "https://download.pytorch.org/torchaudio/models"


def _torch(
    language: str,
    language_name: str,
    model_name: str,
    filename: str,
    *,
    size: str = "约 360MB",
) -> AlignmentModelSpec:
    model_id = "whisperx-align-en-large" if language == "en" else f"whisperx-align-{language}"
    return AlignmentModelSpec(
        id=model_id,
        language=language,
        language_name=language_name,
        model_name=model_name,
        source="torchaudio",
        filename=filename,
        url=f"{_TORCHAUDIO_BASE}/{filename}",
        size=size,
    )


def _hf(language: str, language_name: str, repo: str, *, size: str = "") -> AlignmentModelSpec:
    return AlignmentModelSpec(
        id=f"whisperx-align-{language}",
        language=language,
        language_name=language_name,
        model_name=repo,
        source="huggingface",
        size=size,
    )


ALIGNMENT_MODELS: tuple[AlignmentModelSpec, ...] = (
    _torch(
        "en",
        "英语",
        "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "wav2vec2_fairseq_large_lv60k_asr_ls960.pth",
        size="1.18GB",
    ),
    _torch("fr", "法语", "VOXPOPULI_ASR_BASE_10K_FR", "wav2vec2_voxpopuli_base_10k_asr_fr.pt"),
    _torch("de", "德语", "VOXPOPULI_ASR_BASE_10K_DE", "wav2vec2_voxpopuli_base_10k_asr_de.pt"),
    _torch("es", "西班牙语", "VOXPOPULI_ASR_BASE_10K_ES", "wav2vec2_voxpopuli_base_10k_asr_es.pt"),
    _torch("it", "意大利语", "VOXPOPULI_ASR_BASE_10K_IT", "wav2vec2_voxpopuli_base_10k_asr_it.pt"),
    _hf("ja", "日语", "jonatasgrosman/wav2vec2-large-xlsr-53-japanese", size="约 1.2GB"),
    _hf("zh", "中文", "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", size="约 1.2GB"),
    _hf("nl", "荷兰语", "jonatasgrosman/wav2vec2-large-xlsr-53-dutch", size="约 1.2GB"),
    _hf("uk", "乌克兰语", "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm"),
    _hf("pt", "葡萄牙语", "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", size="约 1.2GB"),
    _hf("ar", "阿拉伯语", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", size="约 1.2GB"),
    _hf("cs", "捷克语", "comodoro/wav2vec2-xls-r-300m-cs-250"),
    _hf("ru", "俄语", "jonatasgrosman/wav2vec2-large-xlsr-53-russian", size="约 1.2GB"),
    _hf("pl", "波兰语", "jonatasgrosman/wav2vec2-large-xlsr-53-polish", size="约 1.2GB"),
    _hf("hu", "匈牙利语", "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian", size="约 1.2GB"),
    _hf("fi", "芬兰语", "jonatasgrosman/wav2vec2-large-xlsr-53-finnish", size="约 1.2GB"),
    _hf("fa", "波斯语", "jonatasgrosman/wav2vec2-large-xlsr-53-persian", size="约 1.2GB"),
    _hf("el", "希腊语", "jonatasgrosman/wav2vec2-large-xlsr-53-greek", size="约 1.2GB"),
    _hf("tr", "土耳其语", "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish"),
    _hf("da", "丹麦语", "saattrupdan/wav2vec2-xls-r-300m-ftspeech"),
    _hf("he", "希伯来语", "imvladikon/wav2vec2-xls-r-300m-hebrew"),
    _hf("vi", "越南语", "nguyenvulebinh/wav2vec2-base-vi-vlsp2020", size="约 360MB"),
    _hf("ko", "韩语", "kresnik/wav2vec2-large-xlsr-korean", size="约 1.2GB"),
    _hf("ur", "乌尔都语", "kingabzpro/wav2vec2-large-xls-r-300m-Urdu"),
    _hf("te", "泰卢固语", "anuragshas/wav2vec2-large-xlsr-53-telugu", size="约 1.2GB"),
    _hf("hi", "印地语", "theainerd/Wav2Vec2-large-xlsr-hindi", size="约 1.2GB"),
    _hf("ca", "加泰罗尼亚语", "softcatala/wav2vec2-large-xlsr-catala", size="约 1.2GB"),
    _hf("ml", "马拉雅拉姆语", "gvs/wav2vec2-large-xlsr-malayalam", size="约 1.2GB"),
    _hf("no", "挪威语（书面语）", "NbAiLab/nb-wav2vec2-1b-bokmaal-v2"),
    _hf("nn", "新挪威语", "NbAiLab/nb-wav2vec2-1b-nynorsk"),
    _hf("sk", "斯洛伐克语", "comodoro/wav2vec2-xls-r-300m-sk-cv8"),
    _hf("sl", "斯洛文尼亚语", "anton-l/wav2vec2-large-xlsr-53-slovenian", size="约 1.2GB"),
    _hf("hr", "克罗地亚语", "classla/wav2vec2-xls-r-parlaspeech-hr"),
    _hf("ro", "罗马尼亚语", "gigant/romanian-wav2vec2"),
    _hf("eu", "巴斯克语", "stefan-it/wav2vec2-large-xlsr-53-basque", size="约 1.2GB"),
    _hf("gl", "加利西亚语", "ifrz/wav2vec2-large-xlsr-galician", size="约 1.2GB"),
    _hf("ka", "格鲁吉亚语", "xsway/wav2vec2-large-xlsr-georgian", size="约 1.2GB"),
    _hf("lv", "拉脱维亚语", "jimregan/wav2vec2-large-xlsr-latvian-cv", size="约 1.2GB"),
    _hf("tl", "菲律宾语", "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official"),
    _hf("sv", "瑞典语", "KBLab/wav2vec2-large-voxrex-swedish"),
    _hf("id", "印度尼西亚语", "cahya/wav2vec2-large-xlsr-indonesian", size="约 1.2GB"),
)

ALIGNMENT_MODEL_BY_ID = {model.id: model for model in ALIGNMENT_MODELS}
ALIGNMENT_MODEL_BY_LANGUAGE = {model.language: model for model in ALIGNMENT_MODELS}
ALIGNMENT_MODEL_BY_NAME = {model.model_name: model for model in ALIGNMENT_MODELS}

LANGUAGE_ALIASES = {"nb": "no", "yue": "zh"}


def normalize_alignment_language(language: str | None) -> str:
    code = (language or "").strip().lower().replace("_", "-").split("-", 1)[0]
    return LANGUAGE_ALIASES.get(code, code)


def alignment_model_for_language(language: str | None) -> AlignmentModelSpec | None:
    return ALIGNMENT_MODEL_BY_LANGUAGE.get(normalize_alignment_language(language))


def alignment_model_path(spec: AlignmentModelSpec, models_dir: str | Path) -> Path:
    """Return the managed on-disk location for an alignment model."""
    root = Path(models_dir).expanduser()
    if spec.source == "torchaudio":
        return root / spec.filename
    return root / f"models--{spec.model_name.replace('/', '--')}"


def is_alignment_model_ready(spec: AlignmentModelSpec, models_dir: str | Path) -> bool:
    """Check whether a managed alignment model is complete enough to load offline."""
    path = alignment_model_path(spec, models_dir)
    if spec.source == "torchaudio":
        return path.is_file() and path.stat().st_size > 0

    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        return False
    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            continue
        if any(
            (snapshot / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        ):
            return True
    return False
