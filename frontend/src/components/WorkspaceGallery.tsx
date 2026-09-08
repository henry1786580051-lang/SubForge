"use client";

import { useState } from "react";
import { InspectorDisclosure } from "./InspectorDisclosure";
import { Panel, TaskActionCard, ToggleLine } from "./WorkspaceControls";

/** Development-only fixture; never mounts in the packaged application. */
export function WorkspaceGallery() {
  const [open, setOpen] = useState(false);
  const [checked, setChecked] = useState(true);
  if (process.env.NODE_ENV !== "development") return null;
  return <>
    <button className="fixed bottom-2 left-2 z-50 rounded-lg bg-surface px-3 py-2 text-xs" onClick={() => setOpen(!open)}>控件检查</button>
    {open && <div role="dialog" aria-label="开发控件检查" className="fixed inset-8 z-40 overflow-auto bg-background p-6" onKeyDown={(event) => { if (event.key === "Escape") setOpen(false); }}>
      <button className="mb-4" onClick={() => setOpen(false)}>关闭检查</button>
      <div className="flex flex-wrap gap-6">
        {[260, 312].map((width) => <aside key={width} className="workspace-inspector" style={{ width, minWidth: width }}>
          <div className="inspector-scroll">
            <Panel title={`${width} pt · 控件状态`} icon="solar:tuning-square-2-bold-duotone">
              <div className="inspector-rows">
                <ToggleLine label="正常开关" checked={checked} onChange={setChecked} />
                <ToggleLine label="禁用状态" description="此项说明需要完整换行，并保持悬停区域与文字高度一致。" checked={false} disabled onChange={() => {}} />
                <label className="inspector-field"><span>源语言</span><select className="input-field"><option>自动检测语言</option><option>English</option></select></label>
              </div>
              <InspectorDisclosure title="模型与凭据 · 长标题覆盖检查"><p>折叠后本区域不应接收键盘焦点。</p><input aria-label="测试凭据" className="input-field" /></InspectorDisclosure>
            </Panel>
            <TaskActionCard title="任务反馈" description="这是受控状态，不会创建真实任务" currentStage="处理中" message="正在处理较长的字幕片段，消息应完整换行显示" disabled={false} running progress={43} onCancel={async () => { await new Promise((resolve) => setTimeout(resolve, 1500)); }} onPrimary={() => {}} primaryLabel="开始" />
          </div>
        </aside>)}
      </div>
    </div>}
  </>;
}
