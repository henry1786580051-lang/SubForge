import { describe, expect, it } from "vitest";
import { modelPresentation } from "./modelPresentation";
const present = (name: string, downloaded = false, state = "missing") => modelPresentation({ name, id: name, downloaded, state });
describe("model presentation", () => {
  it("keeps precision separate from the readable model name", () => {
    expect(present("MLX Large V3 FP16")).toMatchObject({ title: "Whisper Large v3", variant: "FP16" });
  });
  it("preserves version, acceleration and quantization distinctions", () => {
    expect(present("large-v3-turbo").title).toBe("Whisper Large v3 Turbo");
    expect(present("ggml-large-v2-q5_0")).toMatchObject({ title: "Whisper Large v2", variant: "q5_0" });
    expect(present("tiny.en").variant).toBe("en");
  });
  it("does not invent a model family for unknown names", () => {
    expect(present("custom-studio-model").title).toBe("custom-studio-model");
    expect(present("baseball-custom").title).toBe("baseball-custom");
  });
  it("distinguishes local availability from automatic downloads", () => {
    expect(present("base", true, "on_demand").status).toBe("本地可用");
    expect(present("base", false, "on_demand").status).toBe("首次使用自动下载");
    expect(present("base").status).toBe("尚未下载");
  });
});
