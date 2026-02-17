from __future__ import annotations

import argparse
import os
import sys

from .logging_config import setup_logging


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="vc-app client")
	parser.add_argument(
		"--log-level",
		default=None,
		help="Logging level (debug, info, warning, error). Can also use VC_CLIENT_LOG_LEVEL or VC_LOG_LEVEL.",
	)
	parser.add_argument(
		"--server-url",
		default=os.environ.get("VC_SERVER_URL", "ws://127.0.0.1:8765/ws"),
		help="WebSocket signaling URL",
	)
	parser.add_argument(
		"--room",
		default=os.environ.get("VC_ROOM", "default"),
		help="Room to join",
	)
	parser.add_argument(
		"--name",
		default=os.environ.get("VC_NAME", os.environ.get("USER", "")),
		help="Display name",
	)
	args = parser.parse_args(argv)

	setup_logging(args.log_level)

	try:
		from .ui.app import AppConfig, VCClientApp, create_qt_app
	except Exception as e:
		print(f"Failed to import UI dependencies: {e}")
		print("Install client deps with: pip install -r client/requirements.txt")
		return 2

	qt_app = create_qt_app()
	controller = VCClientApp(AppConfig(server_url=args.server_url, room=args.room, name=args.name))
	controller.start()
	qt_app.aboutToQuit.connect(controller.shutdown)

	return qt_app.exec()


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
