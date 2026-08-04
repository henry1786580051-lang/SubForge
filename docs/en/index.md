---
layout: home
title: SubForge - AI Video Subtitle Tool | Free & Open Source
titleTemplate: false
description: Free and open-source AI-powered video subtitle tool. Supports Whisper speech recognition, LLM intelligent segmentation, subtitle optimization, and 99-language translation. Perfect for YouTube, Bilibili, and more.

head:
  - - meta
    - name: keywords
      content: SubForge,video subtitle generator,AI automatic subtitles,Whisper subtitles,LLM subtitle translation,free subtitle tool,open source caption software,video transcription,speech to text,subtitle maker,YouTube subtitle tool,multilingual subtitles,automatic caption generator,subtitle editing software,video captioning,AI subtitle creator,subtitle optimization,video to text converter
  - - meta
    - property: og:title
      content: SubForge - AI Video Subtitle Tool | Free & Open Source
  - - meta
    - property: og:description
      content: Free & open-source AI subtitle tool powered by Whisper & LLM. Supports 99 languages with intelligent segmentation, professional translation, and one-click processing. Perfect for content creators on YouTube, Bilibili, and other platforms.
  - - meta
    - property: og:url
      content: https://henry1786580051-lang.github.io/SubForge/en/
  - - meta
    - property: og:locale
      content: en_US
  - - meta
    - property: og:type
      content: website
  - - meta
    - property: article:published_time
      content: 2024-01-01T00:00:00Z
  - - meta
    - property: article:modified_time
      content: 2025-01-25T00:00:00Z
  - - meta
    - name: twitter:title
      content: SubForge - AI Video Subtitle Tool | Free & Open Source
  - - meta
    - name: twitter:description
      content: Free AI-powered subtitle tool with Whisper & LLM. Supports 99 languages, intelligent segmentation, and professional translation. Perfect for content creators.

hero:
  name: SubForge
  text: Professional Video Subtitle Processing
  tagline: Open Source · LLM-Powered · Process 14-minute video in 4 minutes, cost less than $0.002
  image:
    src: /logo.png
    alt: SubForge
  actions:
    - theme: brand
      text: Get Started
      link: /en/guide/getting-started
    - theme: alt
      text: GitHub Repository
      link: https://github.com/henry1786580051-lang/SubForge

features:
  - icon: ⚡
    title: Lightning Fast, Ultra Low Cost
    details: Process 14-minute video in just 4 minutes, cost less than $0.002. Powered by Whisper + LLM stack for quality and speed.

  - icon: 🧠
    title: LLM-Powered Intelligence
    details: Beyond speech recognition. LLM semantic segmentation, auto error correction, terminology unification, expression optimization for professional results.

  - icon: 🌐
    title: Multilingual Support
    details: Recognize 99 languages, translate to 37 languages. Reflection translation mechanism ensures quality with precise timeline alignment.

  - icon: 📖
    title: Fully Open Source & Free
    details: GPL-3.0 license, no hidden fees. Run locally for complete data privacy control. Community-driven continuous improvement.

  - icon: 💻
    title: No High-End Hardware
    details: Run Whisper on CPU, optional GPU acceleration. Choose between cloud API or local offline processing.

  - icon: 📦
    title: Batch Processing
    details: Drag and drop videos for automatic processing with batch queue support. From recognition to translation to synthesis, zero manual intervention.

  - icon: 🎨
    title: Professional Subtitle Styles
    details: Built-in style templates. Support hard/soft subtitles with SRT/ASS/VTT multi-format output.

  - icon: 🔧
    title: Advanced Features
    details: VAD voice detection, vocal separation, word-level timestamps, manuscript matching, custom prompts, and more.

  - icon: 🖥️
    title: Cross-Platform Desktop App
    details: Windows and macOS desktop installers use a Next.js interface with a local FastAPI backend, real-time preview, and subtitle editing. Linux remains available through the CLI and source installation.
---

## Interface Preview

<div align="center">
  <img src="https://h1.appinn.me/file/1731487405884_main.png" alt="Software Interface Preview" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
</div>

## Quick Experience

::: code-group

```bash [Windows]
# Download and run the installer directly
# Or run from source
git clone https://github.com/henry1786580051-lang/SubForge.git
cd SubForge
run.bat
```

```bash [macOS/Linux]
# Use automatic installation script
git clone https://github.com/henry1786580051-lang/SubForge.git
cd SubForge
chmod +x run.sh
./run.sh
```

:::

## Why Choose SubForge?

- **🎯 Efficient Processing**: Full processing of a 14-minute video takes only about 4 minutes, costing less than ¥0.01
- **🌟 Quality Assurance**: Uses advanced Whisper models and large language models to ensure subtitle quality
- **💡 Intelligent Optimization**: Automatically corrects typos, unifies terminology, optimizes expressions
- **🚀 Easy to Use**: Drag and drop videos for fully automatic processing, no complex configuration needed

## Core Features

### Speech Recognition & Transcription

- Supports Whisper API, FasterWhisper, WhisperCpp, and other engines
- Supports 99 language recognition
- Supports VAD (Voice Activity Detection)
- Supports vocal separation

### Intelligent Subtitle Processing

- LLM semantic segmentation for smoother reading
- Automatically optimizes terminology, code snippets, mathematical formulas
- Supports manuscript matching to improve accuracy
- Precise subtitle timeline alignment

### High-Quality Translation

- Supports LLM translation, Google Translate, Microsoft Azure Translator, and DeepLX
- Reflection translation mechanism improves translation quality
- Maintains complete timeline consistency
- Supports 37 target languages

### Video Synthesis

- Supports hard subtitles and soft subtitles
- Rich subtitle style templates
- Supports multiple subtitle formats (SRT, ASS, VTT, TXT)
- Supports batch video processing

## Get Started

Ready to begin? Check out the [Getting Started Guide](/en/guide/getting-started) to learn how to use SubForge.

<style>
:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
</style>
