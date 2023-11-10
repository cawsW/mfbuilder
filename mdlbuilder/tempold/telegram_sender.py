import requests

def send_to_telegram(message):

    apiToken = '6514208595:AAHq3cPaHm_xej7UO4VA4T2VbT-8I7heNjE'
    chatID = '465441143'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)
