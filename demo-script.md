# Investment Analyst Assistant - SC Demo 演示脚本

## 演示环境准备

- 确保应用已部署并可通过 CloudFront URL 访问
- 已在 Amazon Cognito 创建测试用户
- 已上传测试用的 10K/10Q 财务文档到 Knowledge Base

---

## 场景1：股票基本面分析

### 输入
```
股票代码: AAPL
```

### 操作步骤
1. 登录应用
2. 选择"股票基本面分析"功能
3. 输入股票代码 `AAPL`
4. 点击分析

### 预期输出
- 收入报表数据（Revenue, Net Income, EPS等）
- 图表展示财务趋势
- LLM 生成的财务摘要，包含：
  - 营收增长分析
  - 利润率变化
  - 关键财务指标解读

### 演示要点
- 展示 LangChain Agent 如何调用外部 API 获取实时财务数据
- 强调 Amazon Nova Pro 模型对结构化数据的分析能力
- 展示多种数据可视化形式（图表、表格、文字摘要）

---

## 场景2：财务文档 RAG 问答

### 输入
```
问题: What are the main risk factors mentioned in the 10K report?
```

### 操作步骤
1. 确保已上传公司 10K 文档
2. 选择"文档问答"功能
3. 输入上述问题
4. 查看回答及引用来源

### 预期输出
- 基于 10K 文档的风险因素摘要
- 引用原文段落（Citations）
- 相关性评分

### 演示要点
- 展示 Amazon Bedrock Knowledge Base 的 RAG 能力
- 强调 Amazon Titan Embeddings 的向量化处理
- 展示 OpenSearch Serverless 的相似度搜索
- 突出回答中的引用功能，确保可追溯性

---

## 场景3：新闻情绪分析

### 输入
```
股票代码: AAPL
分析类型: 新闻情绪
```

### 操作步骤
1. 选择"新闻情绪分析"功能
2. 输入股票代码 `AAPL`
3. 触发 Bedrock Agent 执行分析

### 预期输出
- 近期新闻列表
- 情绪分类（正面/负面/中性）
- 综合情绪摘要
- 投资建议参考

### 演示要点
- 展示 Bedrock Agent 调用 Lambda 获取实时新闻（Alpha Vantage API）
- 强调 LLM 对非结构化新闻数据的情绪分析能力
- 展示 Amazon Bedrock Guardrails 如何过滤不当投资建议

---

## 架构亮点总结

| 组件 | 作用 |
|------|------|
| Amazon Nova Pro | 核心 LLM，执行分析和摘要 |
| Amazon Titan Embeddings | 文档向量化 |
| OpenSearch Serverless | 向量存储和相似度搜索 |
| Bedrock Knowledge Base | RAG 文档检索 |
| Bedrock Agent | 编排工具调用 |
| Bedrock Guardrails | 内容安全过滤 |

## 注意事项

- 演示时强调这是 Guidance 示例，非生产就绪代码
- 提醒观众实际投资决策需要专业判断
- 展示 Guardrails 如何阻止生成具体投资建议
