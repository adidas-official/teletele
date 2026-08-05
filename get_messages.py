import os
import sys
import asyncio
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from telethon import TelegramClient
from datetime import datetime, timedelta

# Načtení proměnných ze souboru .env
load_dotenv()

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")

# Kontrola, zda byly klíče v .env nalezeny
if not API_ID or not API_HASH:
    print("[-] Chyba: V .env souboru chybí API_ID nebo API_HASH.")
    sys.exit(1)

# Přetypování API_ID na číslo
API_ID = int(API_ID)


def load_targets_from_file(file_path):
    targets = []
    with open(file_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            target = line.strip()
            if not target or target.startswith('#'):
                continue

            try:
                targets.append(int(target))
            except ValueError:
                targets.append(target)

    return targets


def safe_filename(value):
    value = str(value).strip()
    safe_chars = []

    for char in value:
        if char.isalnum() or char in ('-', '_', '.'):
            safe_chars.append(char)
        else:
            safe_chars.append('_')

    cleaned = ''.join(safe_chars).strip('._')
    return cleaned or 'messages'

async def download_messages(client, target, today=False, limit=None, specific_date=None):
    messages_data = []

    try:
        entity = await client.get_entity(target)
        channel_name = getattr(entity, 'username', None) or getattr(entity, 'title', None) or target

        # Určení cílového data před začátkem cyklu
        target_date = datetime.now().date() if today else specific_date

        async for message in client.iter_messages(target, limit=limit):
            if not message.date:
                continue

            msg_date = message.date.date()

            if target_date:
                if msg_date > target_date:
                    continue  # Zpráva je novější než požadované datum -> přeskočit
                if msg_date < target_date:
                    break     # Zpráva je starší -> jsme za cílovým dnem, ukončit cyklus

            messages_data.append({
                "id": message.id,
                "date": str(message.date),
                "sender_id": message.sender_id,
                "text": message.text or "",
                "reply_to_msg_id": message.reply_to_msg_id if message.reply_to else None,
                "views": message.views,
                "forwards": message.forwards
            })

        output_file = f"messages_export/messages_{safe_filename(channel_name)}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=4)

        return messages_data, channel_name, output_file

    except Exception as e:
        print(f"[-] Chyba při stahování pro cíl {target}: {e}")
        return [], None, None

async def main():
    parser = argparse.ArgumentParser(description="Download recent Telegram messages to JSON.")
    parser.add_argument(
        "target",
        nargs="?",
        default="-10093848934",
        help="Target chat/channel ID or username",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=100,
        help="Number of last messages to download",
    )
    parser.add_argument(
        "-g",
        "--groups-file",
        help="Path to a text file with one group ID, channel ID, or username per line",
    )
    parser.add_argument(
        "-t",
        "--today",
        action="store_true",
        default=False,
        help="If set, only download messages from today"
    )
    parser.add_argument(
        "-d",
        "--date",
        type=str,
        help="Download messages from a specific date (format: YYYY-MM-DD)"
    )
    args = parser.parse_args()

    if args.groups_file:
        targets = load_targets_from_file(args.groups_file)
        if not targets:
            print(f"[-] Chyba: Soubor '{args.groups_file}' neobsahuje žádné cíle.")
            sys.exit(1)
    else:
        target_raw = args.target

        # Přetypování na int, pokud jde o číselné ID
        try:
            target = int(target_raw)
        except ValueError:
            target = target_raw

        targets = [target]

    async with TelegramClient('osint_session', API_ID, API_HASH) as client:
        if args.today:
            download_log_message = f"[+] Připojuji se a stahuji zprávy z dnešního dne pro cíle: {', '.join(str(t) for t in targets)}..."

            for target in targets:
                print(download_log_message)
                messages_data, channel_name, output_file = await download_messages(client, target, today=True)
                after_download_log_message = f"[+] Hotovo! Staženo {len(messages_data)} zpráv z '{channel_name}' do souboru '{output_file}'."
                print(after_download_log_message)

        elif args.date:
            
            download_log_message = f"[+] Připojuji se a stahuji zprávy z {args.date} pro cíle: {', '.join(str(t) for t in targets)}..."

            try:
                specific_date = datetime.strptime(args.date, "%Y-%m-%d").date()
            except ValueError:
                print("[-] Chyba: Datum musí být ve formátu YYYY-MM-DD.")
                sys.exit(1)

            for target in targets:
                print(download_log_message)
                messages_data, channel_name, output_file = await download_messages(client, target, specific_date=specific_date)
                after_download_log_message = f"[+] Hotovo! Staženo {len(messages_data)} zpráv z '{channel_name}' do souboru '{output_file}'."
                print(after_download_log_message)

        else:
            download_log_message = f"[+] Připojuji se a stahuji posledních {args.limit} zpráv pro cíle: {', '.join(str(t) for t in targets)}..."

            for target in targets:
                print(download_log_message)
                messages_data, channel_name, output_file = await download_messages(client, target, limit=args.limit)
                after_download_log_message = f"[+] Hotovo! Staženo {len(messages_data)} zpráv z '{channel_name}' do souboru '{output_file}'."
                print(after_download_log_message)


if __name__ == '__main__':
    asyncio.run(main())
