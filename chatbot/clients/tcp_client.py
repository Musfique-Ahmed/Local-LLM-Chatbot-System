"""Simple CLI TCP client for manual testing of the chatbot server."""

import json
import socket

from chatbot import config


def main() -> None:
    """Run the interactive chat loop until the user quits."""
    user_id = input("user_id: ").strip()
    if not user_id:
        print("user_id is required")
        return

    sock = socket.create_connection((config.HOST, config.PORT))
    sock_file = sock.makefile("rwb", buffering=0)

    try:
        while True:
            try:
                message = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not message:
                continue
            if message.lower() == "quit":
                break

            payload_msg = "__clear__" if message.lower() == "clear" else message
            payload = json.dumps(
                {"user_id": user_id, "message": payload_msg}
            ) + "\n"
            sock_file.write(payload.encode("utf-8"))

            line = sock_file.readline()
            if not line:
                print("(server closed connection)")
                break
            try:
                reply = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError:
                print(f"(malformed response) {line!r}")
                continue

            if "error" in reply:
                print(f"error: {reply['error']}")
            else:
                print(f"bot> {reply.get('response', '')}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
