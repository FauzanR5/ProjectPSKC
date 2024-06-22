from controller import Robot
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

MAX_SPEED = 6.28

# Nama sensor inframerah depan
infrared_sensors_names = ["ps7", "ps6", "ps0", "ps1"]

robot = Robot()

time_step = int(robot.getBasicTimeStep())

# Mengaktifkan sensor inframerah
infrared_sensors = [robot.getDevice(name) for name in infrared_sensors_names]
for sensor in infrared_sensors:
    sensor.enable(time_step)

# Dapatkan motor dan atur posisi ke tak terhingga (kontrol kecepatan)
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Inisialisasi error, delta error, dan variabel PID
previous_errors = [0] * len(infrared_sensors_names)
integral_errors = [0] * len(infrared_sensors_names)
target_value = 100  # Nilai target untuk sensor

# Konstanta PID
Kp = 0.6
Ki = 0.01
Kd = 0.1

# Fuzzy logic setup
sensor_0 = ctrl.Antecedent(np.arange(-100, 101, 1), 'sensor_0')
sensor_1 = ctrl.Antecedent(np.arange(-100, 101, 1), 'sensor_1')
sensor_6 = ctrl.Antecedent(np.arange(-100, 101, 1), 'sensor_6')
sensor_7 = ctrl.Antecedent(np.arange(-100, 101, 1), 'sensor_7')
output = ctrl.Consequent(np.arange(-6.28, 6.29, 1), 'output')

# Membership functions
for sensor in [sensor_0, sensor_1, sensor_6, sensor_7]:
    sensor['far'] = fuzz.trapmf(sensor.universe, [-100, -100, -50, 0])
    sensor['medium'] = fuzz.trimf(sensor.universe, [-50, 0, 50])
    sensor['close'] = fuzz.trapmf(sensor.universe, [0, 50, 100, 100])

output['turn_left'] = fuzz.trimf(output.universe, [-6.28, -3.14, 0])
output['straight'] = fuzz.trimf(output.universe, [-3.14, 0, 3.14])
output['turn_right'] = fuzz.trimf(output.universe, [0, 3.14, 6.28])

# Rule Base
rules = []
conditions = ['far', 'medium', 'close']
for s0 in conditions:
    for s1 in conditions:
        for s6 in conditions:
            for s7 in conditions:
                if s7 == 'close' or s6 == 'close':
                    output_state = 'turn_left'
                elif s0 == 'close' or s1 == 'close':
                    output_state = 'turn_right'
                else:
                    output_state = 'straight'
                rules.append(ctrl.Rule(sensor_0[s0] & sensor_1[s1] & sensor_6[s6] & sensor_7[s7], output[output_state]))

# Membuat kontrol sistem
fuzzy_control = ctrl.ControlSystem(rules)
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_control)

# Pilih metode kendali
USE_PID = False # True jika menggunakan PID dan False jika menggunakan Fuzzy
USE_FUZZY = not USE_PID

while robot.step(time_step) != -1:
    # Membaca nilai sensor inframerah
    sensor_values = [sensor.getValue() for sensor in infrared_sensors]

    # Hitung error dan delta error untuk setiap sensor
    sensors = [value - target_value for value in sensor_values]
    delta_sensors = [sensors[i] - previous_errors[i] for i in range(len(sensors))]

    # Perbaharui integral error
    integral_errors = [integral_errors[i] + sensors[i] for i in range(len(sensors))]

    if USE_PID:
        # Kendali PID untuk setiap sensor
        pid_outputs = [
            Kp * sensors[i] + Ki * integral_errors[i] + Kd * delta_sensors[i]
            for i in range(len(sensors))
        ]

        # Rata-rata output PID untuk kecepatan motor
        left_speed = MAX_SPEED - sum(pid_outputs[:2]) / 2
        right_speed = MAX_SPEED - sum(pid_outputs[2:]) / 2
        left_speed = min(MAX_SPEED, max(-MAX_SPEED, left_speed))
        right_speed = min(MAX_SPEED, max(-MAX_SPEED, right_speed))

        # Cetak sensor, delta sensor, dan kecepatan motor
        print(f"Sensors: {sensors}, Delta Sensors: {delta_sensors}, DeltaSpeed Left: {left_speed}, DeltaSpeed Right: {right_speed}")

    elif USE_FUZZY:
        # Fuzzy logic control
        fuzzy_sim.input['sensor_7'] = sensors[0]
        fuzzy_sim.input['sensor_6'] = sensors[1]
        fuzzy_sim.input['sensor_0'] = sensors[2]
        fuzzy_sim.input['sensor_1'] = sensors[3]

        fuzzy_sim.compute()
        fuzzy_output = fuzzy_sim.output['output']

        # Kecepatan motor berdasarkan kendali fuzzy
        left_speed = MAX_SPEED - fuzzy_output * MAX_SPEED
        right_speed = MAX_SPEED + fuzzy_output * MAX_SPEED
        left_speed = min(MAX_SPEED, max(-MAX_SPEED, left_speed))
        right_speed = min(MAX_SPEED, max(-MAX_SPEED, right_speed))

        # Cetak sensor dan kecepatan motor
        print(f"Sensors: {sensors}, Fuzzy Output: {fuzzy_output}, Left Speed: {left_speed}, Right Speed: {right_speed}")

    # Atur kecepatan motor
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    # Simpan error sensor untuk iterasi berikutnya
    previous_errors = sensors.copy()
