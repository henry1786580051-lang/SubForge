# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
ROOT = os.path.dirname(os.path.abspath(SPEC))

# MLX is injected after PyInstaller finishes. Importing mlx_whisper while
# collecting submodules initializes Metal and can terminate headless builds.
optional_hiddenimports = []
optional_datas = []

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
# transcription and WhisperX only for forced alignment; diarization and the full
# Transformers model zoo are not needed and make the macOS bundle much larger.
optional_hiddenimports += [
    'whisperx',
    'whisperx.alignment',
    'whisperx.audio',
    'whisperx.schema',
    'whisperx.log_utils',
    'whisperx.utils',
    'transformers',
    'transformers.models.wav2vec2',
    'transformers.models.wav2vec2.modeling_wav2vec2',
    'transformers.models.wav2vec2.processing_wav2vec2',
        'transformers.models.wav2vec2.tokenization_wav2vec2',
        'numba',
        'llvmlite',
        'tokenizers',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
]
for optional_pkg in ('whisperx', 'transformers', 'tokenizers', 'tiktoken'):
    try:
        optional_datas += collect_data_files(optional_pkg, include_py_files=False)
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

# Also include backend source files as data (for import at runtime)
backend_src = []
backend_dir = os.path.join(ROOT, 'backend', 'app')
if os.path.isdir(backend_dir):
    for root, dirs, files in os.walk(backend_dir):
        for f in files:
            if f.endswith('.py'):
                src = os.path.join(root, f)
                dest = os.path.relpath(root, os.path.join(ROOT, 'backend'))
                backend_src.append((src, dest))

# Collect subforge prompt/resource files
vc_datas = collect_data_files('subforge.core.prompts', include_py_files=False)

# Collect resource directory (assets, fonts, subtitle_style)
resource_datas = []
resource_dir = os.path.join(ROOT, 'resource')
if os.path.isdir(resource_dir):
    for root, dirs, files in os.walk(resource_dir):
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
    binaries=[],
    datas=frontend_datas + backend_datas + vc_datas + backend_src + resource_datas + runtime_datas + optional_datas,
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
        'subforge.core.asr.faster_whisper',
        'subforge.core.asr.chunked_asr',
        'subforge.core.asr.transcribe',
        'subforge.core.asr.base',
        'subforge.core.asr.asr_data',
        'subforge.core.asr.content_integrity',
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
        'torchvision',
        'mlx', 'mlx_whisper',
        'modelscope',
        'tensorflow', 'keras',
        'PyQt5', 'PyQt-Fluent-Widgets', 'qfluentwidgets',
        'matplotlib',
        'IPython', 'jupyter',
        'test', 'tests',
        'whisperx.diarize',
        'whisperx.asr', 'whisperx.transcribe', 'whisperx.vads',
        'whisperx.vads.pyannote', 'whisperx.vads.silero',
        'pyannote', 'pyannote.audio', 'pyannote.core', 'pyannote.database',
        'pyannoteai', 'pyannoteai_sdk',
        'pyannote.metrics', 'pyannote.pipeline',
        'torchcodec',
        'lightning', 'pytorch_lightning',
        'sklearn', 'scikit_learn',
        'optuna',
    ],
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
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
    },
)
