from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
load_dotenv()


long_text = """
In recent decades, the world has undergone rapid technological transformation that has reshaped how people live, work, and communicate.
As cities expand and populations grow, urban infrastructure faces increasing pressure to provide sustainable housing, transportation, and energy solutions.
Governments around the globe are investing in renewable energy sources to reduce dependence on fossil fuels and mitigate environmental damage.
However, critics argue that policy changes often move too slowly to address the urgency of climate-related challenges.
Meanwhile, researchers continue to develop innovative technologies aimed at improving efficiency and lowering carbon emissions.
One major challenge is balancing economic growth with environmental responsibility in developing nations.
Another significant factor influencing global stability is the widening gap between wealthy and disadvantaged communities.
Education systems are evolving to equip younger generations with digital skills and critical thinking abilities.
Businesses are also adapting by implementing flexible work models and embracing automation to remain competitive.
Consumers, too, are becoming more conscious of the social and ecological impact of their purchasing decisions.
Despite these efforts, misinformation and political polarization can hinder collective progress.
Climate change remains a central issue, intensifying natural disasters and disrupting food and water supplies.
International cooperation is essential for creating comprehensive strategies that transcend national borders.
Looking ahead, experts emphasize the importance of innovation, resilience, and ethical leadership.
Ultimately, the choices societies make today will determine the quality of life for future generations.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50,
)

parts = splitter.create_documents([long_text])

# for part in parts:
    # print(part.page_content)
    # print("-"*30)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

chain_sumarize = load_summarize_chain(llm, chain_type="stuff", verbose=False)

result = chain_sumarize.invoke({"input_documents": parts})
print(result['output_text'])