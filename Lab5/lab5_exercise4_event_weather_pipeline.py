import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

# Step 1: Extract Event Details
class EventDetails(BaseModel):
    event_type: str
    event_date: str
    location: str
    indoor_outdoor_preference: str

parser_step1 = PydanticOutputParser(pydantic_object=EventDetails)
step1_prompt = PromptTemplate(
    template="""Extract event details from the description.
{format_instructions}
Description: {event_description_text}""",
    input_variables=["event_description_text"],
    partial_variables={"format_instructions": parser_step1.get_format_instructions()}
)
step1_chain = step1_prompt | llm | parser_step1

# Step 2: Weather Response Normalizer
class WeatherData(BaseModel):
    temperature: str
    precipitation_chance: str
    conditions: str

parser_step2 = PydanticOutputParser(pydantic_object=WeatherData)
step2_prompt = PromptTemplate(
    template="""Normalize this raw weather API response into structured weather data.
{format_instructions}
API Payload: {raw_api_payload}""",
    input_variables=["raw_api_payload"],
    partial_variables={"format_instructions": parser_step2.get_format_instructions()}
)
step2_chain = step2_prompt | llm | parser_step2

# Step 3: Assistant Recommendation
step3_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an event planning assistant. Recommend whether to proceed as planned, reschedule, or move indoors based on event details and weather conditions."),
    ("user", """Event Details:
- Type: {event_type}
- Date: {event_date}
- Location: {location}
- Preference: {indoor_outdoor_preference}

Weather Forecast:
- Temperature: {temperature}
- Precipitation: {precipitation_chance}
- Conditions: {conditions}

Provide the final recommendation message to the event organizer.""")
])
step3_chain = step3_prompt | llm | StrOutputParser()

if __name__ == "__main__":
    event_desc = "We are hosting an outdoor college graduation party on Saturday, October 24 in Pune. Outdoor setup is strongly preferred."
    
    event_info = step1_chain.invoke({"event_description_text": event_desc})
    print("--- Step 1: Event Details ---")
    print(event_info.model_dump())

    # Mock raw API response for the location and date
    raw_weather = "{'loc': 'Pune', 'forecast': {'temp_c': 29, 'rain_prob': '75%', 'sky': 'heavy thunderstorms'}}"
    weather_info = step2_chain.invoke({"raw_api_payload": raw_weather})
    print("\n--- Step 2: Normalized Weather ---")
    print(weather_info.model_dump())

    merged_inputs = {**event_info.model_dump(), **weather_info.model_dump()}
    recommendation = step3_chain.invoke(merged_inputs)
    print("\n--- Step 3: Recommendation Message ---")
    print(recommendation)