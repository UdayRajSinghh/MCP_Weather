from flask import Flask, request, jsonify, render_template
import requests
import json
import tiktoken
import os
from datetime import datetime

app = Flask(__name__)

# MCP Config
MAX_HISTORY_MESSAGES = 10  # Maximum number of messages to keep in history
TOKEN_LIMIT = 2048  # Context window for Llama 3.2:1B
SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions thoughtfully and concisely.
Be friendly and informative.
When providing weather information, use the data that was retrieved for you.
You are having a realtime weather data api to help you with that."""
MODEL_NAME = "llama3.2:1b"  # Fixed to Llama 3.2:1B

# OpenWeatherMap Config
WEATHER_API_KEY = "3632ae6d6d94c669625518029df0b8ad"  # Replace with your actual API key
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Initialize tokenizer for token counting
encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Count the number of tokens in a text string"""
    return len(encoder.encode(text))

class MCPManager:
    def __init__(self, system_prompt, max_tokens):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages = []
        self.system_prompt_tokens = count_tokens(system_prompt)
        self.current_tokens = self.system_prompt_tokens

    def add_message(self, role, content):
        """Add a message to the conversation history with MCP management"""
        message = {"role": role, "content": content}
        message_tokens = count_tokens(content) + 4  # 4 tokens for role metadata

        # Check if adding this message would exceed token limit
        if self.current_tokens + message_tokens > self.max_tokens:
            # Prune older messages until we have space
            self._prune_messages(message_tokens)
        
        self.messages.append(message)
        self.current_tokens += message_tokens
        return self.get_context()

    def _prune_messages(self, tokens_needed):
        """Remove oldest messages to make room for new ones"""
        while (self.current_tokens + tokens_needed > self.max_tokens and 
               len(self.messages) > 0):
            # Remove the oldest message (but never the system prompt)
            removed_message = self.messages.pop(0)
            removed_tokens = count_tokens(removed_message["content"]) + 4
            self.current_tokens -= removed_tokens

    def get_context(self):
        """Get the current context formatted for Ollama"""
        # Always include system prompt first
        formatted_messages = [{"role": "system", "content": self.system_prompt}]
        # Then add conversation history
        formatted_messages.extend(self.messages[-MAX_HISTORY_MESSAGES:])
        return formatted_messages

# Weather related functions
def is_weather_request(message):
    """Check if the message is asking about weather"""
    weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny", "cloudy", 
                        "how hot", "how cold", "how's the weather", "what's the weather"]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in weather_keywords)

def extract_location(message):
    """Extract location from the message - this is a simple implementation
    In a production app, you might want to use NLP or a more sophisticated approach"""
    message_lower = message.lower()
    
    # Looking for patterns like "weather in X", "weather at X", "temperature in X", etc.
    location_patterns = ["weather in ", "weather at ", "temperature in ", "temperature at ",
                         "forecast in ", "forecast at ", "how's the weather in ", 
                         "what's the weather in ", "how is the weather in "]
    
    for pattern in location_patterns:
        if pattern in message_lower:
            # Extract everything after the pattern
            location_start = message_lower.find(pattern) + len(pattern)
            location = message_lower[location_start:].strip()
            # Remove any trailing punctuation or words like "today", "tomorrow", etc.
            end_markers = ["?", ".", "!", "today", "tomorrow", "now", "right now"]
            for marker in end_markers:
                if f" {marker}" in location:
                    location = location.split(f" {marker}")[0]
                if location.endswith(marker):
                    location = location[:-len(marker)]
            return location.strip()
    
    # If no pattern is found, return None
    return None

def get_weather_data(location):
    """Get weather data from OpenWeatherMap API"""
    try:
        params = {
            "q": location,
            "appid": WEATHER_API_KEY,
            "units": "metric"  # For temperature in Celsius
        }
        response = requests.get(WEATHER_BASE_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Format the weather data
            weather_info = {
                "location": f"{data['name']}, {data.get('sys', {}).get('country', '')}",
                "description": data.get('weather', [{}])[0].get('description', 'Unknown'),
                "temperature": data.get('main', {}).get('temp', 'Unknown'),
                "feels_like": data.get('main', {}).get('feels_like', 'Unknown'),
                "humidity": data.get('main', {}).get('humidity', 'Unknown'),
                "wind_speed": data.get('wind', {}).get('speed', 'Unknown'),
                "timestamp": datetime.utcfromtimestamp(data.get('dt', 0)).strftime('%Y-%m-%d %H:%M:%S UTC')
            }
            
            return weather_info, None
        else:
            error_msg = f"Error retrieving weather data: {response.json().get('message', 'Unknown error')}"
            return None, error_msg
    
    except Exception as e:
        return None, f"Exception occurred: {str(e)}"

def format_weather_data(weather_info):
    """Format weather data for inclusion in the LLM context"""
    if not weather_info:
        return "Weather data could not be retrieved."
    
    formatted_data = f"""
WEATHER DATA (retrieved at {weather_info['timestamp']}):
Location: {weather_info['location']}
Current conditions: {weather_info['description']}
Temperature: {weather_info['temperature']}°C
Feels like: {weather_info['feels_like']}°C
Humidity: {weather_info['humidity']}%
Wind speed: {weather_info['wind_speed']} m/s
"""
    return formatted_data

# Initialize the MCP Manager
mcp = MCPManager(SYSTEM_PROMPT, TOKEN_LIMIT)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Add user message to MCP context
    mcp.add_message("user", user_message)
    
    # Check if this is a weather request
    weather_data_message = None
    if is_weather_request(user_message):
        location = extract_location(user_message)
        if location:
            weather_info, error = get_weather_data(location)
            if weather_info:
                weather_data_message = format_weather_data(weather_info)
                # Add weather data as an "system" message to the context
                mcp.add_message("system", f"The following is current weather information: {weather_data_message}")
            else:
                # Add error as system message
                mcp.add_message("system", f"Weather information request failed: {error}")
    
    # Format messages for Ollama
    messages = mcp.get_context()
    
    # Call Ollama API
    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            assistant_message = response.json().get('message', {}).get('content', '')
            # Add assistant response to MCP context
            mcp.add_message("assistant", assistant_message)
            return jsonify({"response": assistant_message})
        else:
            return jsonify({"error": f"Ollama API error: {response.text}"}), response.status_code
    
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)