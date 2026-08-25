import { afterEach, describe, expect, it, vi } from "vitest";

import { subtitlesApi, tasksApi } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("desktop API client", () => {
  it("surfaces backend detail for failed task requests", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "task not found" }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        })
      )
    );

    await expect(tasksApi.get("missing")).rejects.toThrow("task not found");
  });

  it("rejects malformed JSON instead of returning an invalid task", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("not-json", { status: 200 }))
    );

    await expect(tasksApi.get("task-1")).rejects.toThrow("Invalid JSON response");
  });

  it("posts the edited subtitle snapshot when saving", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: "saved", file_path: "/tmp/out.srt", count: 1 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);
    const segments = [
      {
        id: 1,
        start: "00:00:00.000",
        end: "00:00:01.000",
        text: "source",
        translated: "译文",
      },
    ];

    await subtitlesApi.save("/tmp/out.srt", segments);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/subtitles/save",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ file_path: "/tmp/out.srt", segments }),
      })
    );
  });

  it("returns binary export content without JSON parsing", async () => {
    const expected = new Blob(["subtitle"]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(expected, { status: 200 }))
    );

    const result = await subtitlesApi.exportPost([], "srt", "bilingual", "result.srt");

    expect(await result.text()).toBe("subtitle");
  });
});
