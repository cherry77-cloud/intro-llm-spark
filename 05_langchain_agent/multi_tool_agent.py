import os
import requests
from typing import Optional
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Environment Setup
load_dotenv()

# ----------------------------
# Image Generation Tool
# ----------------------------
def generate_image(prompt: str) -> str:
    """Generate image from text using ModelsLab API.
    Input: Text description of desired image
    Returns: Path to saved image or error message"""
    api_key = os.getenv("MODELSLAB_API_KEY")
    if not api_key:
        return "Error: MODELSLAB_API_KEY not set"
        
    response = requests.post(
        "https://modelslab.com/api/v6/realtime/text2img",
        json={
            "key": api_key,
            "prompt": prompt,
            "width": 512,
            "height": 512
        },
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "success" and data.get("output"):
            img_url = data["output"][0]
            img_data = requests.get(img_url).content
            img_path = "generated_image.png"
            with open(img_path, "wb") as f:
                f.write(img_data)
            return f"Image generated successfully at: {img_path}"
    return f"Error: {response.text[:200]}"

# ----------------------------
# Geolocation Tool
# ----------------------------
def get_geolocation(ip: str = "") -> str:
    """Get location by IP using IP-API"""
    url = f"http://ip-api.com/json/{ip}" if ip else "http://ip-api.com/json/"
    data = requests.get(url).json()
    return f"Location: {data['city']}, {data['country']}" if data['status'] == 'success' else "Location unavailable"

# ----------------------------
# Weather Tool
# ----------------------------
def get_weather(location: str = "") -> str:
    """Get weather for a location using OpenWeatherMap API"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY not set"
        
    if not location:
        # Get current location first
        location_data = requests.get("http://ip-api.com/json/").json()
        if location_data['status'] == 'success':
            location = f"{location_data['city']},{location_data['countryCode']}"
        else:
            return "Could not determine location"
            
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    data = requests.get(url).json()
    
    if data.get("main"):
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        weather = data["weather"][0]["description"]
        return f"Weather in {location}: {weather}, Temperature: {temp}°C, Humidity: {humidity}%"
    return f"Weather data unavailable for {location}"

# ----------------------------
# System Initialization
# ----------------------------
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# Tool Assembly
tools = load_tools(["llm-math"], llm=llm)

tools.extend([
    Tool(
        name="image_generator",
        func=generate_image,
        description="Generates images from text prompts. Input: Description of desired image"
    ),
    Tool(
        name="geolocation",
        func=get_geolocation,
        description="Finds location by IP. Input: IP address or empty for current location"
    ),
    Tool(
        name="weather",
        func=get_weather,
        description="Gets weather for a location. Input: City name or empty for current location"
    ),
    TavilySearchResults(
        name="web_search",
        description="Searches online information",
        max_results=2,
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    ),
    PythonREPLTool(
        name="python_repl",
        description="Executes Python code in sandbox",
        sanitize_input=True,
        globals_whitelist=["__builtins__", "math", "json"]
    )
])

# Agent Configuration
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# ----------------------------
# Demonstration
# ----------------------------
if __name__ == "__main__":
    demo_queries = [
        "Generate an image of a futuristic city at night",
        "What's my current location?",
        "What's the weather in BeiJing?",
        "Find the latest news about Mars exploration",
        "Calculate 15% of 880",
        "What's the location of IP 8.8.8.8?",
        "Tom M. Mitchell is an American computer scientis and the Founders University Professor at Carnegie Mellon University (CMU) what book did he write?"
    ]
    
    for query in demo_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        result = agent.invoke(query)
        print("Response:", result)
