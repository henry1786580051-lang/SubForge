# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
ROOT = os.path.dirname(os.path.abspath(SPEC))

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

a = Analysis(
    [os.path.join(ROOT, 'launcher.py')],
    pathex=[ROOT, os.path.join(ROOT, 'backend')],
    binaries=[],
    datas=frontend_datas + backend_datas + vc_datas + backend_src,
    hiddenimports=[
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
        'subforge.core.translate',
        'subforge.core.split',
        'subforge.core.optimize',
        'subforge.core.subtitle',
        'subforge.core.llm',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchaudio',
        'transformers', 'tokenizers',
        'modelscope',
        'tensorflow', 'keras',
        'PyQt5', 'PyQt-Fluent-Widgets', 'qfluentwidgets',
        'matplotlib', 'scipy',
        'IPython', 'jupyter',
        'test', 'tests', 'unittest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
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
    name='Subtitle.app',
    icon=None,
    bundle_identifier='com.subtitle.web',
    info_plist={
        'CFBundleDisplayName': 'Subtitle',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
    },
)
