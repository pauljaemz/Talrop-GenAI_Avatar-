from typing import List, Dict, Any

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


QA_SYSTEM_PROMPT = """You are an educational assistant and patient teacher.
Explain concepts thoroughly and break down complex ideas into simpler parts.
Use examples and analogies when helpful.
Use only the following pieces of retrieved context to answer the question.
If the answer is not contained within the retrieved context, say you don't know.

Context:
{context}"""


CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore, k: int = 5):
    """
    Builds the LCEL RAG chain using:
      - HuggingFaceEndpoint (SmolLM3-3B)
      - retriever from provided vectorstore
      - teacher-style QA_SYSTEM_PROMPT (single “personality” here)
    """
    endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceTB/SmolLM3-3B",
        temperature=0.7,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=endpoint)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    def contextualized_question(input_dict: Dict[str, Any]) -> str:
        if input_dict.get("chat_history"):
            return contextualize_q_chain.invoke(input_dict)
        return input_dict["input"]

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def retrieve_and_format(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        question = contextualized_question(input_dict)
        docs = retriever.invoke(question)
        return {
            "context": _format_docs(docs),
            "input": input_dict["input"],
            "chat_history": input_dict.get("chat_history", []),
            "source_docs": docs,
        }

    rag_chain = (
        RunnableLambda(retrieve_and_format)
        | RunnablePassthrough.assign(
            answer=lambda x: (qa_prompt | llm | StrOutputParser()).invoke({
                "context": x["context"],
                "input": x["input"],
                "chat_history": x["chat_history"],
            })
        )
    )
    return rag_chain
