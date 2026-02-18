from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .widgets import LogPanel, PeerList, StatusCard


class MainWindow(QtWidgets.QMainWindow):
	connect_clicked = QtCore.Signal()
	disconnect_clicked = QtCore.Signal()
	join_clicked = QtCore.Signal()
	leave_clicked = QtCore.Signal()

	def __init__(self):
		super().__init__()
		self.setWindowTitle("vc-app")

		central = QtWidgets.QWidget()
		self.setCentralWidget(central)

		self.server_url_edit = QtWidgets.QLineEdit()
		self.server_url_edit.setPlaceholderText("ws://host:8765/ws")

		self.name_edit = QtWidgets.QLineEdit()
		self.name_edit.setPlaceholderText("Display name")

		self.room_edit = QtWidgets.QLineEdit()
		self.room_edit.setPlaceholderText("Room")

		self.mic_combo = QtWidgets.QComboBox()
		self.vad_threshold_spin = QtWidgets.QSpinBox()
		self.vad_threshold_spin.setRange(0, 20000)
		self.vad_threshold_spin.setSingleStep(50)
		self.vad_threshold_spin.setToolTip("Audio activity threshold (RMS). Lower = more sensitive")
		self.speaker_combo = QtWidgets.QComboBox()

		self.connect_btn = QtWidgets.QPushButton("Connect")
		self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
		self.join_btn = QtWidgets.QPushButton("Join")
		self.leave_btn = QtWidgets.QPushButton("Leave")

		self.peer_list = PeerList()
		self.log_panel = LogPanel()
		self.status_card = StatusCard()

		form = QtWidgets.QFormLayout()
		form.addRow("Server", self.server_url_edit)
		form.addRow("Name", self.name_edit)
		form.addRow("Room", self.room_edit)
		mic_row = QtWidgets.QHBoxLayout()
		mic_row.setContentsMargins(0, 0, 0, 0)
		mic_row.addWidget(self.mic_combo, 1)
		mic_row.addWidget(QtWidgets.QLabel("Thr"))
		mic_row.addWidget(self.vad_threshold_spin)
		mic_row_widget = QtWidgets.QWidget()
		mic_row_widget.setLayout(mic_row)
		form.addRow("Mic", mic_row_widget)
		form.addRow("Speaker", self.speaker_combo)

		btn_row = QtWidgets.QHBoxLayout()
		btn_row.addWidget(self.connect_btn)
		btn_row.addWidget(self.disconnect_btn)
		btn_row.addStretch(1)
		btn_row.addWidget(self.join_btn)
		btn_row.addWidget(self.leave_btn)

		left = QtWidgets.QVBoxLayout()
		left.addLayout(form)
		left.addLayout(btn_row)
		left.addWidget(QtWidgets.QLabel("Peers"))
		left.addWidget(self.peer_list, 1)

		right = QtWidgets.QVBoxLayout()
		right.addWidget(QtWidgets.QLabel("Log"))
		right.addWidget(self.log_panel, 1)
		right.addWidget(self.status_card, 0)

		main = QtWidgets.QHBoxLayout(central)
		main.addLayout(left, 1)
		main.addLayout(right, 1)

		self.status = QtWidgets.QStatusBar()
		self.setStatusBar(self.status)
		self.set_status("Idle")

		self.connect_btn.clicked.connect(self.connect_clicked.emit)
		self.disconnect_btn.clicked.connect(self.disconnect_clicked.emit)
		self.join_btn.clicked.connect(self.join_clicked.emit)
		self.leave_btn.clicked.connect(self.leave_clicked.emit)

	def set_status(self, text: str) -> None:
		self.status.showMessage(text)
