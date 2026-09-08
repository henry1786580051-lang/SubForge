import { describe, it, expect, vi } from "vitest";
import { createConfigWriter } from "./configWrites";

describe("configuration write barrier", () => {
  it("serializes writes and waits before a task can read defaults", async () => {
    let release!: () => void;
    const first = new Promise<void>((resolve) => { release = resolve; });
    const write = vi.fn().mockReturnValueOnce(first).mockResolvedValue(undefined);
    const queue = createConfigWriter(write);
    const a = queue.update("model", "small");
    const b = queue.update("model", "large");
    let ready = false;
    const barrier = queue.flush().then(() => { ready = true; });
    await Promise.resolve();
    expect(write).toHaveBeenCalledTimes(1);
    expect(ready).toBe(false);
    release();
    await Promise.all([a, b, barrier]);
    expect(write.mock.calls).toEqual([["model", "small"], ["model", "large"]]);
  });
  it("blocks after failure until that setting is successfully retried", async () => {
    const write = vi.fn().mockRejectedValueOnce(new Error("offline")).mockResolvedValue(undefined);
    const queue = createConfigWriter(write);
    await expect(queue.update("model", "small")).rejects.toThrow("offline");
    await queue.update("language", "en");
    await expect(queue.flush()).rejects.toThrow("配置尚未保存成功");
    await queue.update("model", "small");
    await expect(queue.flush()).resolves.toBeUndefined();
  });
});
