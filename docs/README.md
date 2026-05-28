# SubForge 文档

这是 SubForge 项目的文档源文件，使用 [VitePress](https://vitepress.dev/) 构建。

## 📚 在线查看

文档已自动部署到 GitHub Pages：

**[https://henry1786580051-lang.github.io/SubForge/](https://henry1786580051-lang.github.io/SubForge/)**

## 🚀 本地开发

### 安装依赖

```bash
npm install
```

### 启动开发服务器

```bash
npm run docs:dev
```

访问 http://localhost:5173 查看文档

### 构建文档

```bash
npm run docs:build
```

构建产物位于 `docs/.vitepress/dist/`

### 预览构建结果

```bash
npm run docs:preview
```

## 📁 目录结构

```
docs/
├── .vitepress/
│   ├── config.mts          # VitePress 配置文件（含 SEO 优化）
│   └── theme/              # 自定义主题（可选）
├── public/                 # 静态资源（图片、Logo、robots.txt）
├── guide/                  # 中文使用指南
│   ├── getting-started.md
│   ├── configuration.md
│   └── ...
├── config/                 # 中文配置文档
│   ├── llm.md
│   ├── asr.md
│   └── ...
├── dev/                    # 中文开发者文档
│   ├── architecture.md
│   └── ...
├── en/                     # 英文文档（镜像中文结构）
│   ├── guide/
│   ├── config/
│   └── dev/
└── index.md                # 中文首页
```

## ✍️ 贡献文档

### 添加新页面

1. 在对应目录下创建 Markdown 文件
2. **添加 Frontmatter SEO 优化**（重要！）：

```markdown
---
title: 页面标题 - SubForge
description: 页面描述，包含关键词
head:
  - - meta
    - name: keywords
      content: 关键词1,关键词2,关键词3
---

# 页面标题

内容...
```

3. 在 `.vitepress/config.mts` 的 `sidebar` 中添加链接
4. 提交 PR

### 编辑现有页面

直接编辑 Markdown 文件即可，支持：

- **Markdown 扩展语法**：表格、代码块、提示框等
- **Vue 组件**：可在 Markdown 中使用 Vue 组件
- **自定义容器**：`::: tip`, `::: warning`, `::: danger`

示例：

```md
::: tip 提示
这是一个提示框
:::

::: warning 注意
这是一个警告框
:::

::: danger 危险
这是一个危险警告框
:::
```

### 文档规范

- **文件名**：使用小写字母和连字符（如 `getting-started.md`）
- **标题**：使用清晰的层级结构（# → ## → ###）
- **代码块**：标注语言类型以启用语法高亮
- **图片**：放在 `public/` 目录，使用 `/image.png` 引用
- **链接**：内部链接使用相对路径（如 `/guide/getting-started`）
- **SEO**：每个页面都应添加 title、description 和 keywords

## 🔍 SEO 优化

本文档系统已经过全面 SEO 优化，详情请查看 [SEO_OPTIMIZATION.md](../SEO_OPTIMIZATION.md)。

### 已实施的 SEO 功能

✅ **基础 SEO**

- Title 标签优化
- Meta Description 和 Keywords
- Open Graph（社交媒体卡片）
- Twitter Card
- JSON-LD 结构化数据
- Sitemap 自动生成
- robots.txt
- Canonical URL

✅ **技术 SEO**

- 响应式设计
- Clean URLs
- 快速加载（Vite 优化）
- HTTPS（GitHub Pages）

### 提交到搜索引擎

部署后需要手动提交到搜索引擎：

1. **Google Search Console**
   - 访问 https://search.google.com/search-console
   - 添加网站并验证
   - 提交 sitemap: `https://henry1786580051-lang.github.io/SubForge/sitemap.xml`

2. **Bing Webmaster Tools**
   - 访问 https://www.bing.com/webmasters
   - 添加网站并验证
   - 提交 sitemap

3. **百度站长平台**
   - 访问 https://ziyuan.baidu.com/
   - 添加网站并验证
   - 提交 sitemap

### SEO 检查工具

- [Google PageSpeed Insights](https://pagespeed.web.dev/)
- [Google Rich Results Test](https://search.google.com/test/rich-results)
- [Open Graph Debugger](https://developers.facebook.com/tools/debug/)
- [Twitter Card Validator](https://cards-dev.twitter.com/validator)

## 🌐 多语言支持

文档支持中英双语：

- **中文**：`docs/` 根目录
- **英文**：`docs/en/` 目录

添加新语言：

1. 在 `docs/` 下创建语言目录（如 `ja/`）
2. 在 `.vitepress/config.mts` 中添加 locale 配置
3. 复制文档结构并翻译内容

## 🔧 技术栈

- **VitePress**: 基于 Vite 的静态站点生成器
- **Vue 3**: 组件化开发
- **TypeScript**: 类型安全的配置

## 📝 更新文档

文档更新会自动触发 GitHub Actions 部署：

1. 提交文档修改到 `docs/` 目录
2. 推送到 `master` 或 `main` 分支
3. GitHub Actions 自动构建并部署
4. 约 2-3 分钟后更新生效

## ❓ 常见问题

### 本地开发时看不到样式？

确保已安装依赖：

```bash
npm install
```

### 如何添加自定义样式？

在 `docs/.vitepress/theme/` 目录下创建自定义主题：

```ts
// docs/.vitepress/theme/index.ts
import DefaultTheme from "vitepress/theme";
import "./custom.css";

export default DefaultTheme;
```

### 如何配置搜索功能？

VitePress 默认提供本地搜索，已在 `config.mts` 中配置。

### 如何优化图片？

1. 使用图片压缩工具（如 TinyPNG）
2. 考虑使用 WebP 格式
3. 添加 `loading="lazy"` 属性

### 如何添加 Google Analytics？

在 `config.mts` 的 `head` 中添加：

```typescript
([
  "script",
  {
    async: true,
    src: "https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX",
  },
],
  [
    "script",
    {},
    `
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
`,
  ]);
```

---

更多 VitePress 使用方法请参考 [官方文档](https://vitepress.dev/)。

更多 SEO 优化细节请查看 [SEO_OPTIMIZATION.md](../SEO_OPTIMIZATION.md)。
