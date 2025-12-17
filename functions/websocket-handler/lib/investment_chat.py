import os

import boto3
from aws_lambda_powertools import Logger, Tracer
from langchain.agents import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_aws import ChatBedrock
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_community.chat_message_histories import \
    DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from lib.tools.investment_analysis_tool import (InvestmentAnalysisTool,
                                                get_latest_news,
                                                get_price_history,
                                                get_recommendations,
                                                search_knowledge_base)
from lib.tools.stockPrice import StockPriceTool

logger = Logger(service="investment_analysis")
tracer = Tracer(service="investment_analysis")
import markdown

LLM_MODEL_ID = os.environ["LLM_MODEL_ID"]
KB_ID = os.environ["KB_ID"]
CHAT_HISTORY_TBL_NM = os.environ["CHAT_HISTORY_TBL_NM"]
GUARDRAILS_ID = os.environ["BEDROCK_GUARDRAILSID"]
GUARDRAIL_VERSION = os.environ["BEDROCK_GUARDRAILSVERSION"]

bedrock_region = os.environ["AWS_REGION"]
bedrock_runtime = boto3.client("bedrock-runtime", region_name=bedrock_region)
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=bedrock_region)

logger.info(f"KB_ID : {KB_ID}")

nova_chat_llm = ChatBedrock(
    model_id=LLM_MODEL_ID,
    client=bedrock_runtime,
    model_kwargs={"temperature": 0.2, "top_p": 0.99, "max_tokens": 4096},
    guardrails={"guardrailIdentifier": GUARDRAILS_ID, "guardrailVersion": GUARDRAIL_VERSION},
    disable_streaming=True
)

amzn_kb_retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=KB_ID,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 3}},
)

kb_retriever_tool = create_retriever_tool(
    retriever=amzn_kb_retriever,
    name="knowledge_base",
    description="""This tool provides the historical news related to stock market. 
    Use this tool to get information about sector, to know the key players in a sector
    or to understand the policies and what other companies are doing. 
	""",
)

stock_price = StockPriceTool()
investment_analysis_tool = InvestmentAnalysisTool()
LLM_AGENT_TOOLS = [
    Tool(
        name="search_knowledge_base",
        func=search_knowledge_base,
        description="""This tool provides the historical news related to stock market. 
        Use this tool to get information about sector, to know the key players in a sector
        or to understand the policies and what other companies are doing. 
        """,
    ),
    Tool(
        name="get_price_history",
        func=get_price_history,
        description="""This tool will provide the stock prices of past 6 months.
        The input parameter is stock ticker prices and output will be
        history of end of the day price for past 6 months""",
    ),
    Tool(
        name="get_recommendations",
        func=get_recommendations,
        description="""This tool will provide the recommendations based on the investment analysis.
        The input parameter is stock ticker prices and output will be
        recommendations based on the investment analysis""",
    ),
    Tool(
        name="get_latest_news",
        func=get_latest_news,
        description="""This tool will provide the latest news related to stock market.
        The input parameter is stock ticker prices and output will be
        latest news related to stock market""",
    ),
    Tool(
        name="StockPrice",
        func=stock_price,
        description="Use this tool when you need to retrieve current stock price and historical price.",
    )
]


def chat_investment(user_input, socket_conn_id):
    """
    Chat function that uses RAG pattern - first retrieves from Knowledge Base,
    then generates response with context.
    """
    # Step 1: Retrieve relevant context from Knowledge Base
    logger.info(f"Retrieving context from KB for: {user_input}")
    retrieve_response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": user_input},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": 5}
        }
    )
    
    retrieval_results = retrieve_response.get("retrievalResults", [])
    context_parts = []
    for result in retrieval_results:
        content = result.get("content", {}).get("text", "")
        score = result.get("score", 0)
        source = result.get("location", {}).get("s3Location", {}).get("uri", "Unknown")
        if content:
            context_parts.append(content)
            logger.info(f"KB doc score={score:.2f}, source={source}, content_preview={content[:200]}...")
    
    kb_context = "\n\n".join(context_parts) if context_parts else ""
    logger.info(f"Retrieved {len(retrieval_results)} documents from KB, total context length: {len(kb_context)}")
    
    # Step 2: Build prompt with KB context
    history = DynamoDBChatMessageHistory(
        table_name=CHAT_HISTORY_TBL_NM,
        session_id=socket_conn_id
    )
    
    # Create system prompt that includes KB context
    system_prompt = """You are a helpful investment analyst assistant. 
Answer questions based on the provided context from financial documents.
If the context contains relevant information, use it to provide accurate answers with specific data.
If the context doesn't contain relevant information, say so and provide general knowledge."""
    
    if kb_context:
        system_prompt += f"""

Here is relevant context from the knowledge base documents:
{kb_context}

Use this context to answer the user's question accurately."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | nova_chat_llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="history",
    )
    response = chain_with_history.invoke({"question": user_input}, {"configurable": {"session_id": socket_conn_id}})
    logger.info(f"chat response: {response.content}")
    return markdown.markdown(response.content)


def query_knowledge_base_rag(user_query: str, socket_conn_id: str) -> str:
    """
    Query Knowledge Base using RAG (Retrieve and Generate) pattern.
    This function retrieves relevant documents from KB and uses LLM to generate answer.
    """
    logger.info(f"RAG Query: {user_query}")
    
    # Step 1: Retrieve relevant documents from Knowledge Base
    retrieve_response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": user_query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": 5}
        }
    )
    
    # Extract retrieved content
    retrieval_results = retrieve_response.get("retrievalResults", [])
    context_parts = []
    citations = []
    
    for i, result in enumerate(retrieval_results):
        content = result.get("content", {}).get("text", "")
        source = result.get("location", {}).get("s3Location", {}).get("uri", "Unknown")
        score = result.get("score", 0)
        context_parts.append(f"[Document {i+1}] (Score: {score:.2f})\n{content}")
        citations.append({"source": source, "score": score})
    
    context = "\n\n".join(context_parts)
    logger.info(f"Retrieved {len(retrieval_results)} documents from Knowledge Base")
    
    if not context_parts:
        return {"answer": "No relevant documents found in the knowledge base.", "citations": []}
    
    # Step 2: Generate answer using LLM with retrieved context
    rag_prompt = f"""Based on the following context from financial documents, please answer the user's question.
If the answer cannot be found in the context, say so clearly.

Context:
{context}

User Question: {user_query}

Please provide a comprehensive answer based on the context above. Include specific numbers and data from the documents when available."""

    response = nova_chat_llm.invoke(rag_prompt)
    answer = response.content
    
    logger.info(f"RAG response generated successfully")
    
    return {
        "answer": markdown.markdown(answer),
        "citations": citations,
        "retrieved_count": len(retrieval_results)
    }