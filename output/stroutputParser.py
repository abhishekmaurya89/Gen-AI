from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Write a story on {topic}',
    input_variables=['topic']
)
template2 = PromptTemplate(
    template='Write a summary of /n {text}',
    input_variables=['text']
)
prompt1 = template1.invoke({'topic':'snake'})
result = model.invoke(prompt1)
prompt2 = template2.invoke({'text':result.content})
result1 = model.invoke(prompt2)
print(result1.content)
