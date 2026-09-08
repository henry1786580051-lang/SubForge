"use client";

import { useId, useState, type ReactNode } from "react";

/** A single uninterrupted expansion; hidden controls leave keyboard navigation. */
export function InspectorDisclosure({ title, children, open = false }: {
  title: ReactNode;
  children: ReactNode;
  open?: boolean;
}) {
  const id = useId();
  const [choice, setChoice] = useState<boolean | null>(null);
  const expanded = choice ?? open;
  return <section className={`inspector-disclosure ${expanded ? "is-expanded" : ""}`}>
    <button type="button" className="inspector-disclosure-trigger" aria-expanded={expanded}
      aria-controls={id} onClick={() => setChoice(!expanded)}>
      <span className="inspector-disclosure-title">{title}</span>
      <svg aria-hidden="true" viewBox="0 0 16 16" width="14" height="14"><path d="m6 4 4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
    </button>
    <div className="inspector-disclosure-expansion" id={id} inert={!expanded} aria-hidden={!expanded}>
      <div className="inspector-disclosure-clip"><div className="inspector-disclosure-body">{children}</div></div>
    </div>
  </section>;
}
