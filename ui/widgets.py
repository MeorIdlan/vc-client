from __future__ import annotations

from typing import Iterable

from PySide6 import QtCore, QtWidgets


class LogPanel(QtWidgets.QPlainTextEdit):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setReadOnly(True)
		self.setMaximumBlockCount(2000)

	@QtCore.Slot(str)
	def append_log(self, message: str) -> None:
		self.appendPlainText(message)


class PeerList(QtWidgets.QListWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

	def set_peers(self, peers: Iterable[str]) -> None:
		self.clear()
		for p in peers:
			self.addItem(str(p))
