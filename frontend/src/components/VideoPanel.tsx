"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { useAppStore } from "@/store/appStore";
import { filesApi, API_BASE } from "@/lib/api";
import { formatTime, formatDuration, formatSize, parseSrtTime } from "@/lib/format";

export function VideoPanel() {
  const { videoFile, setVideoFile, fileInfo, setFileInfo, subtitles, currentTime, setCurrentTime, isPlaying, setIsPlaying, seekToTime, setSeekToTime } = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const [urlInput, setUrlInput] = useState("");
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [duration, setDuration] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!videoFile || !videoFile.startsWith("/")) return;
    filesApi.info(videoFile).then(setFileInfo).catch(() => {});
  }, [videoFile, setFileInfo]);

  // Seek video when subtitle double-clicked
  useEffect(() => {
    if (seekToTime === null || !videoRef.current) return;
    videoRef.current.currentTime = seekToTime;
    setSeekToTime(null);
  }, [seekToTime, setSeekToTime]);

  const handleFile = useCallback(async (file: File) => {
    setIsUploading(true);
    try {
      const { file_path } = await filesApi.upload(file);
      setVideoFile(file_path);
    } catch (err) { useAppStore.getState().setError(err instanceof Error ? err.message : "上传失败"); } finally { setIsUploading(false); }
  }, [setVideoFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); }, []);
  const handleDragLeave = useCallback(() => { setIsDragging(false); }, []);
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false);
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);
  const handleFileSelect = useCallback(() => { fileInputRef.current?.click(); }, []);
  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) handleFile(e.target.files[0]);
  }, [handleFile]);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play().then(() => setIsPlaying(true)).catch(() => {});
    } else {
      video.pause();
      setIsPlaying(false);
    }
  }, [setIsPlaying]);

  const seek = useCallback((seconds: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + seconds));
  }, []);

  const handleSeek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const video = videoRef.current;
    if (!video || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    video.currentTime = ((e.clientX - rect.left) / rect.width) * duration;
  }, [duration]);

  const hasVideo = videoFile && videoFile.startsWith("/") && fileInfo?.video;
  const thumbnailUrl = useMemo(
    () => (videoFile && videoFile.startsWith("/") ? filesApi.thumbnailUrl(videoFile) : null),
    [videoFile]
  );

  // Pre-parse subtitle times for performance
  const parsedSubtitles = useMemo(() =>
    subtitles.map((s) => ({
      ...s,
      startSec: parseSrtTime(s.start),
      endSec: parseSrtTime(s.end),
    })),
    [subtitles]
  );

  // Find current subtitle for overlay
  const currentSubtitle = useMemo(() =>
    parsedSubtitles.find((s) => currentTime >= s.startSec && currentTime <= s.endSec),
    [parsedSubtitles, currentTime]
  );

  return (
    <div className="flex flex-col h-full bg-surface relative">
      <input ref={fileInputRef} type="file" accept="video/*,audio/*" className="hidden" onChange={handleFileChange} />

      {/* Video area */}
      <div
        className={`flex-1 flex items-center justify-center relative transition-all duration-300 ${isDragging ? "drop-zone-active" : ""}`}
        onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}
      >
        {isUploading ? (
          <div className="text-center">
            <div className="w-10 h-10 mx-auto mb-3 rounded-full border-2 border-accent/20 border-t-accent animate-spin" />
            <p className="text-sm text-text-secondary">上传中...</p>
          </div>
        ) : !videoFile ? (
          <div
            className={`w-[85%] max-w-2xl aspect-video rounded-xl border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${
              isDragging
                ? "border-accent/50 bg-accent-dim"
                : "border-border hover:border-accent/30 hover:bg-accent-dim drop-zone-idle"
            }`}
            onClick={handleFileSelect}
          >
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 transition-all duration-300 ${isDragging ? "bg-accent-dim scale-110" : "bg-gradient-to-br from-[rgba(0,0,0,0.02)] to-[rgba(0,0,0,0.04)]"}`}>
              <svg className={`w-8 h-8 transition-colors ${isDragging ? "text-accent" : "text-text-muted"}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <p className="text-sm text-text-secondary mb-1">{isDragging ? "松开以导入文件" : "拖拽视频或音频文件到此处"}</p>
            <p className="text-xs text-text-muted">支持 MP4, MKV, AVI, MOV, MP3, WAV 等格式</p>
          </div>
        ) : (
          <div className="w-[85%] max-w-2xl aspect-video rounded-xl bg-black border border-border flex items-center justify-center relative overflow-hidden shadow-md">
            {hasVideo && (
              <video
                ref={videoRef}
                src={`${API_BASE}/api/files/stream?path=${encodeURIComponent(videoFile)}`}
                className="absolute inset-0 w-full h-full object-contain"
                onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime || 0)}
                onLoadedMetadata={() => setDuration(videoRef.current?.duration || 0)}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => setIsPlaying(false)}
                onError={(e) => {
                  const v = e.target as HTMLVideoElement;
                  v.style.opacity = "0.3";
                  v.style.pointerEvents = "none";
                  useAppStore.getState().setError("视频加载失败，请检查文件格式或网络连接");
                }}
              />
            )}
            {/* Subtitle overlay */}
            {currentSubtitle && isPlaying && (
              <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20 max-w-[90%] text-center pointer-events-none">
                <div className="inline-block bg-black/75 text-white text-sm px-3 py-1.5 rounded-md backdrop-blur-sm leading-relaxed shadow-lg">
                  <span>{currentSubtitle.text}</span>
                  {currentSubtitle.translated && (
                    <><br /><span className="text-white/80">{currentSubtitle.translated}</span></>
                  )}
                </div>
              </div>
            )}
            {thumbnailUrl && !isPlaying && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={thumbnailUrl} className="absolute inset-0 w-full h-full object-contain opacity-30" alt="" />
            )}
            {!isPlaying && (
              <div className="text-center relative z-10 cursor-pointer" onClick={togglePlay}>
                <div className="w-16 h-16 mx-auto mb-3 rounded-full bg-white/10 flex items-center justify-center hover:bg-white/20 transition-colors backdrop-blur-sm">
                  <svg className="w-8 h-8 text-white/80" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
                </div>
                <p className="text-sm text-white/60 truncate max-w-xs">{videoFile.split("/").pop() || videoFile}</p>
              </div>
            )}
            {fileInfo && (
              <div className="absolute bottom-3 left-3 flex items-center gap-2 z-10">
                {fileInfo.video && <span className="px-2 py-0.5 text-[10px] rounded bg-black/50 text-white/50 backdrop-blur-sm">{fileInfo.video.width}x{fileInfo.video.height}</span>}
                <span className="px-2 py-0.5 text-[10px] rounded bg-black/50 text-white/50 backdrop-blur-sm">{formatDuration(fileInfo.duration)}</span>
                <span className="px-2 py-0.5 text-[10px] rounded bg-black/50 text-white/50 backdrop-blur-sm">{formatSize(fileInfo.size)}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* URL input */}
      {showUrlInput && (
        <div className="px-4 pb-2">
          <div className="flex items-center gap-2 bg-surface rounded-lg border border-border p-1.5 shadow-sm">
            <input type="text" placeholder="输入视频URL (YouTube, Bilibili...)" value={urlInput} onChange={(e) => setUrlInput(e.target.value)}
              className="flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted px-2" />
            <button onClick={() => { if (urlInput) setVideoFile(urlInput); }}
              className="px-3 py-1.5 text-xs rounded-md bg-accent text-white hover:bg-accent-hover transition-all font-medium btn-press">
              下载
            </button>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="px-4 py-3 border-t border-border bg-surface">
        <div className="flex items-center gap-3 mb-3">
          <div className="flex-1 h-1 progress-track cursor-pointer group rounded-full bg-[rgba(0,0,0,0.06)] overflow-hidden" onClick={handleSeek}>
            <div className="progress-bar h-full bg-accent rounded-full transition-all relative" style={{ width: duration > 0 ? `${(currentTime / duration) * 100}%` : "0%" }}>
              <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-accent opacity-0 group-hover:opacity-100 transition-opacity shadow-sm" />
            </div>
          </div>
          <span className="text-[11px] text-text-muted font-mono tabular-nums">
            {formatTime(currentTime)} / {duration > 0 ? formatTime(duration) : fileInfo ? formatDuration(fileInfo.duration) : "00:00"}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <button onClick={() => setShowUrlInput(!showUrlInput)} className="p-1.5 rounded-md text-text-muted hover:text-text-secondary hover:bg-surface-hover transition-all btn-press" title="输入URL">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
            </svg>
          </button>
          <div className="flex items-center gap-3">
            <button onClick={() => seek(-10)} className="text-text-muted hover:text-text-primary transition-colors p-1 btn-press" title="后退10秒">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M6 6h2v12H6zm3.5 6l8.5 6V6z" /></svg>
            </button>
            <button onClick={() => seek(-5)} className="text-text-muted hover:text-text-primary transition-colors p-1 btn-press" title="后退5秒">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M19 12L6 4v16l13-8z" /></svg>
            </button>
            <button onClick={togglePlay}
              className="w-11 h-11 rounded-full bg-accent text-white flex items-center justify-center hover:bg-accent-hover transition-all btn-press shadow-md">
              {isPlaying ? (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" /></svg>
              ) : (
                <svg className="w-5 h-5 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
              )}
            </button>
            <button onClick={() => seek(5)} className="text-text-muted hover:text-text-primary transition-colors p-1 btn-press" title="前进5秒">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M5 12l13-8v16L5 12z" /></svg>
            </button>
            <button onClick={() => seek(10)} className="text-text-muted hover:text-text-primary transition-colors p-1 btn-press" title="前进10秒">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M18 6h-2v12h2zm-3.5 6L6 6v12z" /></svg>
            </button>
          </div>
          <button onClick={handleFileSelect} className="p-1.5 rounded-md text-text-muted hover:text-text-secondary hover:bg-surface-hover transition-all btn-press" title="选择文件">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
