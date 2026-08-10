from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
     model="gemini-3-flash-preview",
)

prompt1=PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is {name}',
    input_variables=['name']
)
prompt = prompt1.invoke({'name':'abhishek'})

result = model.invoke(prompt)

print(result.content)