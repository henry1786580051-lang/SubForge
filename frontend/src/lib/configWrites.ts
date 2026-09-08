/** Order writes and expose a barrier before a task reads persisted defaults. */
export function createConfigWriter(write: (key: string, value: unknown) => Promise<unknown>) {
  let tail: Promise<unknown> = Promise.resolve();
  const failures = new Map<string, unknown>();
  return {
    update(key: string, value: unknown) {
      const pending = tail.then(async () => {
        try { const result = await write(key, value); failures.delete(key); return result; }
        catch (error) { failures.set(key, error); throw error; }
      });
      tail = pending.catch(() => {});
      return pending;
    },
    async flush() {
      let current;
      do { current = tail; await current; } while (current !== tail);
      if (failures.size) throw new Error("配置尚未保存成功，请重试失败的选项后再开始任务。");
    },
  };
}
