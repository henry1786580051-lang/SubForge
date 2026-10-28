import { describe, expect, it } from "vitest";
import { taskReviewSummary } from "./taskReview";

describe("translation recovery and structure review", () => {
  it("distinguishes missing, reviewed and unchecked entries", () => {
    const text = taskReviewSummary({quality_review:{recovery:{status:"partial",missing_indices:[553],reviewed_indices:[1,2,3],unchecked_indices:[554]}}});
    expect(text).toContain("1 条缺译（553）");
    expect(text).toContain("已复核 3 条，未复核 1 条");
  });
  it("shows structural review instead of implying subtitle acceptance", () => {
    expect(taskReviewSummary({quality_review:{structure:{status:"needs_review",unresolved_boundaries:[155,221]}}})).toContain("第 155、221 条附近");
    expect(taskReviewSummary({quality_review:{structure:{status:"passed"}}})).toBe("");
  });
  it("ignores absent or legacy metadata", () => {
    expect(taskReviewSummary(undefined)).toBe("");
    expect(taskReviewSummary({quality_review:null})).toBe("");
  });
});
