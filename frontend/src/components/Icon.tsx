import type { SVGProps } from "react";
import icons from "./icons.json";
/** Bundled Solar vectors (CC BY 4.0). No icon requests or remote fallback. */
export function Icon({ icon, width = 20, height, ...props }: SVGProps<SVGSVGElement> & { icon: string }) {
  const data = (icons as Record<string, { body: string }>)[icon];
  if (!data) return null;
  return <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width={width} height={height ?? width} aria-hidden="true" focusable="false" {...props} dangerouslySetInnerHTML={{ __html: data.body }} />;
}
