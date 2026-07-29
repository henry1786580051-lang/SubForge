export interface ModelCompanyGroup {
  id: string;
  name: string;
  models: string[];
}

const COMPANY_NAMES: Record<string, string> = {
  nvidia: "NVIDIA",
  meta: "Meta",
  "meta-llama": "Meta",
  "deepseek-ai": "DeepSeek",
  deepseek: "DeepSeek",
  mistralai: "Mistral AI",
  mistral: "Mistral AI",
  qwen: "Qwen",
  google: "Google",
  microsoft: "Microsoft",
  openai: "OpenAI",
  moonshotai: "Moonshot AI",
  minimaxai: "MiniMax",
  ai21labs: "AI21 Labs",
  adept: "Adept AI",
  aisingapore: "AI Singapore",
  baai: "BAAI",
  ibm: "IBM",
  writer: "Writer",
  upstage: "Upstage",
  tiiuae: "TII",
  databricks: "Databricks",
  snowflake: "Snowflake",
  "01-ai": "01.AI",
  baichuan: "Baichuan",
  "baichuan-inc": "Baichuan",
  rakuten: "Rakuten",
  mediatek: "MediaTek",
  thudm: "THUDM",
  zyphra: "Zyphra",
  sarvamai: "Sarvam AI",
  naver: "NAVER",
  "nv-mistralai": "NVIDIA × Mistral AI",
  poolside: "Poolside",
  "stepfun-ai": "StepFun",
  stockmark: "Stockmark",
  thinkingmachines: "Thinking Machines",
  "z-ai": "Z.ai",
  bigcode: "BigCode",
  stabilityai: "Stability AI",
};

const COMPANY_ORDER = [
  "nvidia",
  "meta",
  "deepseek-ai",
  "mistralai",
  "qwen",
  "google",
  "microsoft",
  "openai",
  "moonshotai",
  "minimaxai",
];

const COMPANY_ALIASES: Record<string, string> = {
  "meta-llama": "meta",
  deepseek: "deepseek-ai",
  mistral: "mistralai",
  "baichuan-inc": "baichuan",
};

function modelCompanyId(modelId: string): string {
  const normalized = modelId.trim().toLowerCase();
  const prefix = normalized.includes("/") ? normalized.split("/", 1)[0] : "";
  if (prefix) return COMPANY_ALIASES[prefix] || prefix;

  if (normalized.includes("nemotron")) return "nvidia";
  if (normalized.startsWith("llama")) return "meta";
  if (normalized.startsWith("deepseek")) return "deepseek-ai";
  if (normalized.startsWith("mistral") || normalized.startsWith("mixtral")) return "mistralai";
  if (normalized.startsWith("qwen")) return "qwen";
  if (normalized.startsWith("gemma")) return "google";
  if (normalized.startsWith("phi-")) return "microsoft";
  return "other";
}

function formatCompanyName(companyId: string): string {
  const knownName = COMPANY_NAMES[companyId];
  if (knownName) return knownName;
  if (companyId === "other") return "其他模型";
  return companyId
    .split(/[-_]/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function groupNvidiaModels(models: string[], query = ""): ModelCompanyGroup[] {
  const normalizedQuery = query.trim().toLowerCase();
  const groups = new Map<string, string[]>();

  for (const model of models) {
    const cleanModel = model.trim();
    if (!cleanModel || (normalizedQuery && !cleanModel.toLowerCase().includes(normalizedQuery))) {
      continue;
    }
    const companyId = modelCompanyId(cleanModel);
    const companyModels = groups.get(companyId) || [];
    companyModels.push(cleanModel);
    groups.set(companyId, companyModels);
  }

  const order = new Map(COMPANY_ORDER.map((company, index) => [company, index]));
  return Array.from(groups, ([id, companyModels]) => ({
    id,
    name: formatCompanyName(id),
    models: [...new Set(companyModels)].sort((left, right) => left.localeCompare(right)),
  })).sort((left, right) => {
    const leftOrder = order.get(left.id) ?? Number.MAX_SAFE_INTEGER;
    const rightOrder = order.get(right.id) ?? Number.MAX_SAFE_INTEGER;
    return leftOrder - rightOrder || left.name.localeCompare(right.name);
  });
}
