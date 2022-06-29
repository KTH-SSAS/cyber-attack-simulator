import json

import telegram


def notify_ending(message):
    with open("./telegram_keys.json", "r") as keys_file:
        k = json.load(keys_file)
        token = k["telegram_token"]
        chat_id = k["chat_id"]
        bot = telegram.Bot(token=token)
        bot.sendMessage(chat_id=chat_id, text=message)
