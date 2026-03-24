import requests

API_KEY  = "Y0ab13f01fe4b8fa1c959df9305fe3224"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_weather(city):
    try:
        params   = {"q": city, "appid": API_KEY, "units": "metric"}
        response = requests.get(BASE_URL, params=params)
        data     = response.json()

        if data["cod"] != 200:
            return f"Sorry I could not find weather for {city}!"

        city_name = data["name"]
        country   = data["sys"]["country"]
        temp      = data["main"]["temp"]
        feels     = data["main"]["feels_like"]
        humidity  = data["main"]["humidity"]
        desc      = data["weather"][0]["description"].capitalize()
        wind      = data["wind"]["speed"]

        return (
            f"Weather in {city_name}, {country}:\n"
            f"Temperature: {temp}C (feels like {feels}C)\n"
            f"Condition: {desc}\n"
            f"Humidity: {humidity}%\n"
            f"Wind speed: {wind} m/s"
        )
    except Exception as e:
        return "Could not fetch weather. Please check your internet connection!"

if __name__ == "__main__":
    city = input("Enter city name: ")
    print(get_weather(city))