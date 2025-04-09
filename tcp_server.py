import socket
import json

class MarkerPoseServer:
    def __init__(self, host='127.0.0.1', port=1235):
        self.host = host
        self.port = port
        self.client_socket = None
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"[TCP] MarkerPoseServer listening on {self.host}:{self.port}")
        self.client_connected = False

    def wait_for_connection(self):
        self.client_socket, addr = self.server_socket.accept()
        self.client_connected = True
        print(f"[TCP] Connection from {addr}")

    def send_pose(self, marker_id, tvec, rvec):
        if not self.client_connected:
            return

        try:
            payload = {
                "id": int(marker_id),
                "tvec": [float(x) for x in tvec],
                "rvec": [float(r) for r in rvec],
            }
            json_data = json.dumps(payload) + "\n"
            self.client_socket.sendall(json_data.encode('utf-8'))
        except Exception as e:
            print(f"[TCP] Send error: {e}")
            self.client_socket.close()
            self.client_connected = False

    def close(self):
        if self.client_socket:
            self.client_socket.close()
        self.server_socket.close()