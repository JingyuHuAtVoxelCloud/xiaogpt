from __future__ import annotations

import os
import dataclasses
from typing import TYPE_CHECKING, ClassVar

import time
import httpx
from rich import print

from xiaogpt.bot.base_bot import BaseBot, ChatHistoryMixin
from xiaogpt.utils import split_sentences

if TYPE_CHECKING:
    import openai

from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.llms.azure_openai import AzureOpenAI, AsyncAzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings, 
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    SimpleDirectoryReader
    )

@dataclasses.dataclass
class RagBot(ChatHistoryMixin, BaseBot):
    name: ClassVar[str] = "RAG"
    default_options: ClassVar[dict[str, str]] = {"model": "gpt4-1106-prevision"}
    openai_key: str
    api_base: str | None = None
    proxy: str | None = None
    history: list[tuple[str, str]] = dataclasses.field(default_factory=list, init=False)


    def _make_query_engine(self, sess: httpx.AsyncClient, stream=False):

        llm = AzureOpenAI(
            engine="gpt4-1106-prevision",
            api_key=self.openai_key,
            azure_endpoint=self.api_base,
            api_version="2023-12-01-preview",
        )
        embed_model = AzureOpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name="embedding-ada-002-v2",
            api_key=self.openai_key,
            azure_endpoint="http://192.168.12.232:8880",
            api_version="2023-05-15",
        )
        Settings.embed_model = embed_model
        Settings.llm = llm
        # check if storage already exists
        PERSIST_DIR = "xiaogpt/rag/storage"
        if not os.path.exists(PERSIST_DIR):
            # load the documents and create the index
            documents = SimpleDirectoryReader("xiaogpt/rag/data").load_data()
            index = VectorStoreIndex.from_documents(documents)
            # store it for later
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)

        # set Logging to DEBUG for more detailed outputs

        text_qa_template_str = (
            "Context information is"
            " below.\n---------------------\n{context_str}\n---------------------\nUsing"
            " both the context information and also using your own knowledge, answer"
            " the question with less that 100 words: {query_str}\nIf the context isn't helpful, you can also"
            " answer the question on your own.\n"
        )
        text_qa_template = PromptTemplate(text_qa_template_str)

        refine_template_str = (
            "The original question is as follows: {query_str}\nWe have provided an"
            " existing answer: {existing_answer}\nWe have the opportunity to refine"
            " the existing answer (only if needed) with some more context"
            " below.\n------------\n{context_msg}\n------------\nUsing both the new"
            " context and your own knowledge, update existing answer with less than 100 words. \n"
        )
        refine_template = PromptTemplate(refine_template_str)


        query_engine = index.as_query_engine(
            text_qa_template=text_qa_template, 
            refine_template=refine_template,
            llm=llm,
            streaming=stream
            )
        return query_engine
        

    @classmethod
    def from_config(cls, config):
        return cls(
            openai_key=config.openai_key,
            api_base=config.api_base,
            proxy=config.proxy
        )

    async def ask(self, query, **options):
        ms = self.get_messages()
        ms.append({"role": "user", "content": f"{query}"})
        kwargs = {**self.default_options, **options}
        httpx_kwargs = {}
        if self.proxy:
            httpx_kwargs["proxies"] = self.proxy
        async with httpx.AsyncClient(trust_env=True, **httpx_kwargs) as sess:
            query_engine = self._make_query_engine(sess)
            try:
                completion = query_engine.query(query)
            except Exception as e:
                print(str(e))
                return ""
            message = completion.response
            # print(completion.source_nodes[0].get_text())
            self.add_message(query, message)
            print(message)
            return message

    async def ask_stream(self, query, **options):
        ms = self.get_messages()
        ms.append({"role": "user", "content": f"{query}"})
        kwargs = {**self.default_options, **options}
        httpx_kwargs = {}
        if self.proxy:
            httpx_kwargs["proxies"] = self.proxy
        async with httpx.AsyncClient(trust_env=True, **httpx_kwargs) as sess:
            query_engine = self._make_query_engine(sess, stream=True)
            try:
                completion = query_engine.query(query)
            except Exception as e:
                print(str(e))
                return

            async def text_gen():
                async for event in completion:
                    if not event.response:
                        continue
                    chunk_message = event.response
                    if chunk_message.response is None:
                        continue
                    print(chunk_message.response, end="")
                    yield chunk_message.response

            message = ""
            try:
                async for sentence in split_sentences(text_gen()):
                    message += sentence
                    yield sentence
            finally:
                print()
                self.add_message(query, message)



import functools
import dataclasses
from typing import Any, AsyncIterator, Literal, Optional

@dataclasses.dataclass
class Config:
    openai_key: str = "voxelcloud"
    proxy: str | None = None
    api_base: str = "http://192.168.12.232:8881"
    stream: bool = False
    bot: str = "chatgptapi"
    gpt_options: dict[str, Any] = dataclasses.field(default_factory=dict)




import asyncio
async def main():
    config = Config()  # 假设 Config 类已经定义并可以接受默认参数
    bot = RagBot.from_config(config)
    # 询问问题
    response = await bot.ask("什么是光疗？")
    
    print(response)

# 运行异步 main 函数
if __name__ == "__main__":
    asyncio.run(main())
    