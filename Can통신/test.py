import can
import pickle
import numpy as np

# HEX 값을 로그 값으로 변환
def hex_to_log(x):
    try:
        if isinstance(x, str):
            return np.log1p(float(int(x, 16)))
        else:
            return 0
    except ValueError:
        return 0

# 모델 로드 함수
def load_model(model_path='cb_model.pkl'):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# 모터 정지 함수
def stop_motor():
    print("Motor stopped!")

# CAN 메시지 전처리
def preprocess_can_message(message):
    id_value = hex_to_log(hex(message.arbitration_id))
    data_values = message.data.apply(hex_to_log)
    return [message.timestamp, len(message.data), id_value, data_values]  #

# 예측 및 모터 제어
def predict_and_control(model, message):
    data = preprocess_can_message(message)
    prediction = model.predict([data])
    print(f"Prediction: {prediction[0]}")
    
    if prediction[0] == 1:  # 예: 1이 "모터 정지"를 의미
        stop_motor()

# CAN 데이터 수신 및 처리
def receive_can_data_and_predict(model, interface='can0', timeout=None):
    try:
        bus = can.interface.Bus(channel=interface, bustype='socketcan')
        print(f"Listening on {interface}...")
        while True:
            message = bus.recv(timeout=timeout)
            if message:
                message.data = list(message.data)
                print(f"Timestamp: {message.timestamp}, ID: {hex(message.arbitration_id)}, Data: {message.data}")
                predict_and_control(model, message)
            else:
                print("Timeout reached or no messages received.")
                break
    except can.CanError as e:
        print(f"CAN interface error: {e}")
    finally:
        print("Finished listening.")

if __name__ == "__main__":
    cb_model = load_model('cb_model.pkl')
    receive_can_data_and_predict(cb_model, interface='vcan0', timeout=10)

