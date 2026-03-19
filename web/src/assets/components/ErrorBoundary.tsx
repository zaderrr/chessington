import { Component, type ReactNode } from "react";

class ErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: { componentStack: string }) {
    console.error("Chessboard error:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return <div style={{ color: "red" }}>Error: {this.state.error?.message}</div>;
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
