"""
serial_reader.py
================
PySerial interface worker for future hardware ESP32 streaming.

Architecture:
- Runs in a background QThread to prevent freezing the GUI.
- Parses telemetry frames sent from ESP32 over USB-UART:
  Expected format: "CSV: sample_idx, raw_x, filtered_x, predicted_class, dominant_freq"
- Handles reconnection and error recovery gracefully.
- Allows Software Test Mode to operate without requiring any connected hardware.
"""

from typing import List, Optional
from PyQt6.QtCore import QThread, pyqtSignal
import serial
import serial.tools.list_ports


def get_available_serial_ports() -> List[str]:
    """Return list of available COM ports on the machine."""
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]


class SerialReaderThread(QThread):
    """
    Background worker for asynchronous serial reading.
    """
    data_packet_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.serial_conn: Optional[serial.Serial] = None

    def connect_serial(self, port: str, baudrate: int = 115200) -> bool:
        self.port = port
        self.baudrate = baudrate
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.connection_status.emit(True, f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            self.connection_status.emit(False, f"Connection failed: {str(e)}")
            return False

    def disconnect_serial(self):
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.connection_status.emit(False, "Disconnected")

    def run(self):
        self.running = True
        while self.running:
            if not self.serial_conn or not self.serial_conn.is_open:
                self.msleep(100)
                continue
                
            try:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                    
                # Example frame: "VIB,raw,filtered,pred_class,dom_freq"
                parts = line.split(',')
                if len(parts) >= 3 and parts[0] == "VIB":
                    packet = {
                        'raw': float(parts[1]),
                        'filtered': float(parts[2]),
                        'class': int(parts[3]) if len(parts) > 3 else 0,
                        'dom_freq': float(parts[4]) if len(parts) > 4 else 0.0
                    }
                    self.data_packet_received.emit(packet)
            except Exception as e:
                self.connection_status.emit(False, f"Serial read error: {str(e)}")
                self.msleep(200)

    def stop(self):
        self.running = False
        self.wait(1000)
