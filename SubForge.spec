# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None
ROOT = os.path.dirname(os.path.abspath(SPEC))
APP_VERSION = os.environ.get('SUBFORGE_BUILD_VERSION', '0.0.0-dev')

# MLX is injected after PyInstaller finishes. Importing mlx_whisper while
# collecting submodules initializes Metal and can terminate headless builds.
optional_hiddenimports = []
optional_datas = []
optional_binaries = []
if os.name == 'nt':
    optional_hiddenimports += [
        'colorama',
        'colorama.ansi',
        'colorama.ansitowin32',
        'colorama.initialise',
        'colorama.win32',
        'colorama.winterm',
        'whisperx.asr',
        'whisperx.vads',
        'whisperx.vads.silero',
        'faster_whisper',
        'ctranslate2',
        'av',
    ]
    for runtime_pkg in ('faster_whisper', 'ctranslate2', 'av', 'whisperx.vads'):
        try:
            optional_hiddenimports += collect_submodules(runtime_pkg)
        except Exception:
            pass
    for runtime_pkg in ('ctranslate2', 'av'):
        try:
            optional_binaries += collect_dynamic_libs(runtime_pkg)
        except Exception:
            pass
    # FasterWhisper loads its packaged Silero ONNX model lazily when VAD starts.
    # Import-only checks pass without it, so collect package data explicitly.
    try:
        optional_datas += collect_data_files('faster_whisper', include_py_files=False)
    except Exception:
        pass

whisperx_runtime_excludes = [] if os.name == 'nt' else [
    'whisperx.asr', 'whisperx.transcribe', 'whisperx.vads',
    'whisperx.vads.pyannote', 'whisperx.vads.silero',
]

# DeepFilterNet is used only for local enhancement inference. Avoid collecting
# training, evaluation, and visualization modules, which pull unnecessary
# dependencies into the desktop bundle.
optional_hiddenimports += [
    'libdf',
    'df',
    'df.enhance',
    'df.checkpoint',
    'df.config',
    'df.deepfilternet',
    'df.deepfilternet2',
    'df.deepfilternet3',
    'df.deepfilternetmf',
    'df.io',
    'df.logger',
    'df.model',
    'df.modules',
    'df.multiframe',
    'df.sepm',
    'df.utils',
    'df.version',
]
for optional_pkg in ('df', 'libdf'):
    try:
        optional_datas += collect_data_files(optional_pkg, include_py_files=False)
    except Exception:
        pass

# Keep WhisperX/Transformers collection narrow. SubForge uses MLX Whisper for
# transcription, WhisperX for forced alignment, and pyannote only for optional
# speaker diarization. Training and visualization stacks remain excluded.
optional_hiddenimports += [
    'whisperx',
    'whisperx.alignment',
    'whisperx.audio',
    'whisperx.schema',
    'whisperx.log_utils',
    'whisperx.utils',
    'pyannote.audio',
    'pyannote.audio.core',
    'pyannote.audio.core.pipeline',
    'pyannote.audio.pipelines',
    'pyannote.audio.pipelines.speaker_diarization',
    'pyannote.audio.models',
    'pyannote.audio.models.segmentation',
    'pyannote.core',
    'pyannote.database',
    'pyannote.metrics',
    'pyannote.pipeline',
    'onnxruntime',
    'onnxruntime.capi._pybind_state',
    'lightning',
    'lightning_fabric',
    'pytorch_lightning',
    'sklearn',
    'transformers',
    'transformers.models.wav2vec2',
    'transformers.models.wav2vec2.modeling_wav2vec2',
    'transformers.models.wav2vec2.processing_wav2vec2',
    'transformers.models.wav2vec2.tokenization_wav2vec2',
    'tokenizers',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
]
try:
    optional_hiddenimports += collect_submodules('pyannote.audio.models')
except Exception:
    pass
for optional_pkg in (
    'whisperx', 'transformers', 'tokenizers', 'tiktoken',
    'pyannote.audio', 'pyannote.core', 'pyannote.database',
    'pyannote.metrics', 'pyannote.pipeline',
):
    try:
        package_datas = collect_data_files(optional_pkg, include_py_files=False)
        if optional_pkg == 'whisperx':
            package_datas = [
                item for item in package_datas
                if os.path.basename(item[0]) != 'pytorch_model.bin'
            ]
        optional_datas += package_datas
    except Exception:
        pass

# Collect frontend static export
frontend_datas = []
frontend_out = os.path.join(ROOT, 'frontend', 'out')
if os.path.isdir(frontend_out):
    for root, dirs, files in os.walk(frontend_out):
        for f in files:
            src = os.path.join(root, f)
            dest = os.path.relpath(root, ROOT)  # "frontend/out/..."
            frontend_datas.append((src, dest))

# Collect backend data files
backend_datas = collect_data_files('app', include_py_files=False)

# Collect subforge prompt/resource files
vc_datas = collect_data_files('subforge.core.prompts', include_py_files=False)

# Collect only resources used by the webview desktop application.
resource_datas = []
resource_dir = os.path.join(ROOT, 'resource')
resource_entries = [
    'assets/en.mp3',
    'assets/logo.png',
    'ten_vad',
]
for entry in resource_entries:
    source = os.path.join(resource_dir, entry)
    if os.path.isfile(source):
        resource_datas.append((source, os.path.dirname(os.path.join('resource', entry))))
    elif os.path.isdir(source):
        for root, dirs, files in os.walk(source):
            dirs.sort()
            files.sort()
            for f in files:
                src = os.path.join(root, f)
                dest = os.path.relpath(root, ROOT)
                resource_datas.append((src, dest))

# Also bundle ffmpeg/ffprobe from desktop-runtime if available
runtime_datas = []
runtime_bin = os.path.join(ROOT, 'build', 'desktop-runtime', 'resource', 'bin')
if os.path.isdir(runtime_bin):
    for f in os.listdir(runtime_bin):
        src = os.path.join(runtime_bin, f)
        if os.path.isfile(src):
            runtime_datas.append((src, 'resource/bin'))

a = Analysis(
    [os.path.join(ROOT, 'launcher.py')],
    pathex=[ROOT, os.path.join(ROOT, 'backend')],
    binaries=optional_binaries,
    datas=frontend_datas + backend_datas + vc_datas + resource_datas + runtime_datas + optional_datas,
    hiddenimports=[
        'torch',
        'torch.hub',
        'webview',
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'pydub',
        'pydub.generators',
        'pydub.utils',
        'fastapi',
        'starlette',
        'starlette.responses',
        'starlette.routing',
        'app.main',
        'app.api.tasks',
        'app.api.transcribe',
        'app.api.subtitle',
        'app.api.config',
        'app.api.websocket',
        'app.api.files',
        'app.api.subtitles',
        'app.api.llm_logs',
        'subforge.core.asr',
        'subforge.core.asr.silero_vad',
        'subforge.core.asr.ten_vad',
        'subforge.core.asr.speech_vad',
        'subforge.core.asr.whisper_api',
        'subforge.core.asr.whisper_cpp',
        'subforge.core.asr.whisperx_asr',
        'subforge.core.asr.speaker_diarization',
        'subforge.core.asr.speaker_embedding_models',
        'subforge.core.asr.speaker_verification',
        'subforge.core.asr.faster_whisper',
        'subforge.core.asr.chunked_asr',
        'subforge.core.asr.transcribe',
        'subforge.core.asr.base',
        'subforge.core.asr.asr_data',
        'subforge.core.translate',
        'subforge.core.split',
        'subforge.core.optimize',
        'subforge.core.subtitle',
        'subforge.core.subtitle.resegment',
        'subforge.core.llm',
        'soundfile',
        'scipy',
        'scipy.io',
        'scipy.io.wavfile',
    ] + optional_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'mlx', 'mlx_whisper',
        'modelscope',
        'tensorflow', 'keras',
        'PyQt5', 'PyQt-Fluent-Widgets', 'qfluentwidgets',
        # Pillow is imported transitively by torchvision during pyannote startup.
        'fontTools', 'edge_tts',
        'matplotlib',
        'IPython', 'jupyter',
        'test', 'tests',
        'whisperx.diarize',
        'pyannoteai', 'pyannoteai_sdk',
        'speechbrain',
        'torchcodec',
        # MLX Whisper's packaged timing module receives a correct pure-Python
        # fallback in build_desktop.py; forced alignment does not use its DTW JIT.
        'numba', 'llvmlite',
        'pytest', '_pytest', 'tkinter',
    ] + whisperx_runtime_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=True,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SubForge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SubForge',
)

app = BUNDLE(
    coll,
    name='SubForge.app',
    icon=os.path.join(ROOT, 'resource', 'assets', 'SubForge.icns'),
    bundle_identifier='com.subforge.app',
    info_plist={
        'CFBundleDisplayName': 'SubForge',
        'CFBundleShortVersionString': APP_VERSION,
        'CFBundleVersion': APP_VERSION,
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
    },
)
