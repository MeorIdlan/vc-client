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


class StatusCard(QtWidgets.QFrame):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
		self.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)

		self._connection = QtWidgets.QLabel("Disconnected")
		self._room = QtWidgets.QLabel("—")
		self._voice = QtWidgets.QLabel("Idle")

		layout = QtWidgets.QVBoxLayout(self)
		layout.setContentsMargins(10, 10, 10, 10)
		layout.setSpacing(6)

		header = QtWidgets.QLabel("Status")
		font = header.font()
		font.setBold(True)
		header.setFont(font)
		layout.addWidget(header)

		form = QtWidgets.QFormLayout()
		form.setContentsMargins(0, 0, 0, 0)
		form.setHorizontalSpacing(12)
		form.setVerticalSpacing(4)
		form.addRow("Connection", self._connection)
		form.addRow("Room", self._room)
		form.addRow("Mic", self._voice)
		layout.addLayout(form)

		# Ensure long values elide rather than expanding too much.
		self._room.setWordWrap(False)
		self._connection.setWordWrap(False)
		self._voice.setWordWrap(False)

	@QtCore.Slot(str)
	def set_connection_state(self, state: str) -> None:
		self._connection.setText(state.strip() or "—")

	@QtCore.Slot(str)
	def set_room(self, room: str) -> None:
		self._room.setText(room.strip() or "—")

	@QtCore.Slot(bool)
	def set_talking(self, talking: bool) -> None:
		self._voice.setText("Talking..." if talking else "Idle")
