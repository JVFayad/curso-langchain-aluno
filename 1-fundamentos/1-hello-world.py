from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
load_dotenv()

# Open AI
#model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)
#message = model.invoke("Hello, world!")

# Gemini
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
message = model.invoke("Hello, world!")

print(message.content)