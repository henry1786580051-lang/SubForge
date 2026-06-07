"use client";
import React from "react";

interface Props { children: React.ReactNode; }
interface State { hasError: boolean; error: Error | null; }

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-dvh items-center justify-center bg-background">
          <div className="text-center p-8 max-w-md">
            <h2 className="text-lg font-semibold text-text-primary mb-2">应用发生错误</h2>
            <p className="text-sm text-text-muted mb-4">{this.state.error?.message}</p>
            <button onClick={() => { this.setState({ hasError: false, error: null }); window.location.reload(); }}
              className="px-4 py-2 bg-accent text-white rounded-md text-sm hover:bg-accent-hover">
              重新加载
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
