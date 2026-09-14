import asyncio
import logging
import os
from dotenv import load_dotenv
from telethon import TelegramClient

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

async def auth():
    client = TelegramClient(
    'osint_session',
    int(os.getenv("API_ID")),
    os.getenv("API_HASH"),
    device_model="Desktop",
    system_version="Linux",
    app_version="4.16.8"
    )
    await client.connect()
    phone = input("Zadejte číslo (+420...): ")
    sent = await client.send_code_request(phone)
    print("Typ odeslání:", type(sent.type).__name__)
    code = input("Zadejte kód: ")
    await client.sign_in(phone, code)
    print("Úspěšně přihlášeno!")
    await client.disconnect()

asyncio.run(auth())