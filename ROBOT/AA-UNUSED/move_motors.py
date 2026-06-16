#!/usr/bin/env python
#
# *********     Gen Write Example      *********
#
#
# Available SCServo model on this example : All models using Protocol SCS
# This example is tested with a SCServo(STS/SMS/SCS), and an URT
# Be sure that SCServo(STS/SMS/SCS) properties are already set as %% ID : 1 / Baudnum : 6 (Baudrate : 1000000)
#

import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
        
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from scservo_sdk import *                    # Uses SCServo SDK library

# Control table address
ADDR_SCS_TORQUE_ENABLE     = 40
ADDR_SCS_GOAL_ACC          = 41
ADDR_SCS_GOAL_POSITION     = 42
ADDR_SCS_GOAL_SPEED        = 46
ADDR_SCS_PRESENT_POSITION  = 56
ADDR_SCS_MIN_POSITION = 36
ADDR_SCS_MAX_POSITION = 38

# Default setting
SCS_ID                      = 1                 # SCServo ID : 1
BAUDRATE                    = 1000000           # SCServo default baudrate : 1000000
DEVICENAME                  = '/dev/tty.usbmodem59700734041'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

SCS_MINIMUM_POSITION_VALUE  = 1900         # SCServo will rotate between this value
SCS_MAXIMUM_POSITION_VALUE  = 3000        # and this value (note that the SCServo would not move when the position value is out of movable range. Check e-manual about the range of the SCServo you use.)
SCS_MOVING_STATUS_THRESHOLD = 20          # SCServo moving status threshold
SCS_MOVING_SPEED            = 300           # SCServo moving speed
SCS_MOVING_ACC              = 50           # SCServo moving acc
protocol_end                = 0           # SCServo bit end(STS/SMS=0, SCS=1)

index = 0
scs_goal_position = [SCS_MINIMUM_POSITION_VALUE, SCS_MAXIMUM_POSITION_VALUE]         # Goal position

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(protocol_end)

# Initialize PortHandler instance
# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
    # Set port baudrate
    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
        # Scanner les IDs des moteurs connectés
        print("Scan des IDs moteurs...")
        ids_found = []
        for test_id in range(1, 11):
            scs_present_position_speed, scs_comm_result, scs_error = packetHandler.read4ByteTxRx(portHandler, test_id, ADDR_SCS_PRESENT_POSITION)
            if scs_comm_result == COMM_SUCCESS and scs_error == 0:
                print(f"Moteur trouvé sur ID {test_id}, position: {SCS_LOWORD(scs_present_position_speed)}")
                ids_found.append(test_id)
        if not ids_found:
            print("Aucun moteur détecté sur les IDs 1 à 10. Vérifiez câblage, alimentation et configuration.")
        else:
            print(f"IDs détectés: {ids_found}")
        # Lire la position du moteur juste après ouverture du port
        scs_present_position_speed, scs_comm_result, scs_error = packetHandler.read4ByteTxRx(portHandler, SCS_ID, ADDR_SCS_PRESENT_POSITION)
        if scs_comm_result == COMM_SUCCESS and scs_error == 0:
            print(f"Position initiale du moteur ID {SCS_ID}: {SCS_LOWORD(scs_present_position_speed)}")
        else:
            print(f"Impossible de lire la position du moteur ID {SCS_ID}. Erreur: {packetHandler.getTxRxResult(scs_comm_result)}")
    else:
        print("Failed to change the baudrate")
        print("Press any key to terminate...")
        getch()
        quit()
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()
    
# Charger le fichier de calibration
import json
calib_path = '/Users/christelle.nollet/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json'
with open(calib_path, 'r') as f:
    calibration = json.load(f)

# Synchroniser les valeurs min/max du fichier de calibration sur chaque moteur
for joint_name, joint_data in calibration.items():
    motor_id = joint_data['id']
    min_val = joint_data['range_min']
    max_val = joint_data['range_max']
    print(f"\nÉcriture min/max pour moteur ID {motor_id} ({joint_name}) : min={min_val}, max={max_val}")
    # Écriture min
    scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, motor_id, ADDR_SCS_MIN_POSITION, min_val)
    if scs_comm_result != COMM_SUCCESS:
        print(f"Erreur écriture min: {packetHandler.getTxRxResult(scs_comm_result)}")
    elif scs_error != 0:
        print(f"Erreur écriture min: {packetHandler.getRxPacketError(scs_error)}")
    else:
        print("Min écrit avec succès.")
    # Écriture max
    scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, motor_id, ADDR_SCS_MAX_POSITION, max_val)
    if scs_comm_result != COMM_SUCCESS:
        print(f"Erreur écriture max: {packetHandler.getTxRxResult(scs_comm_result)}")
    elif scs_error != 0:
        print(f"Erreur écriture max: {packetHandler.getRxPacketError(scs_error)}")
    else:
        print("Max écrit avec succès.")

# Lecture des paramètres internes pour chaque moteur détecté
ADDR_SCS_MODE = 31
ADDR_SCS_MIN_POSITION = 36
ADDR_SCS_MAX_POSITION = 38
for motor_id in ids_found:
    print(f"\nLecture des paramètres du moteur ID {motor_id} :")
    # Mode
    mode, scs_comm_result, scs_error = packetHandler.read1ByteTxRx(portHandler, motor_id, ADDR_SCS_MODE)
    if scs_comm_result == COMM_SUCCESS and scs_error == 0:
        print(f"Mode: {mode}")
    else:
        print(f"Erreur lecture mode: {packetHandler.getTxRxResult(scs_comm_result)}")
    # Min position
    min_pos, scs_comm_result, scs_error = packetHandler.read2ByteTxRx(portHandler, motor_id, ADDR_SCS_MIN_POSITION)
    if scs_comm_result == COMM_SUCCESS and scs_error == 0:
        print(f"Position min: {min_pos}")
    else:
        print(f"Erreur lecture position min: {packetHandler.getTxRxResult(scs_comm_result)}")
    # Max position
    max_pos, scs_comm_result, scs_error = packetHandler.read2ByteTxRx(portHandler, motor_id, ADDR_SCS_MAX_POSITION)
    if scs_comm_result == COMM_SUCCESS and scs_error == 0:
        print(f"Position max: {max_pos}")
    else:
        print(f"Erreur lecture position max: {packetHandler.getTxRxResult(scs_comm_result)}")

# Test: envoyer une position dans la plage min/max à chaque moteur
# for motor_id in ids_found:
#     print(f"\nTest mouvement moteur ID {motor_id} :")
#     # Relire min/max position
#     min_pos, scs_comm_result, scs_error = packetHandler.read2ByteTxRx(portHandler, motor_id, ADDR_SCS_MIN_POSITION)
#     max_pos, scs_comm_result2, scs_error2 = packetHandler.read2ByteTxRx(portHandler, motor_id, ADDR_SCS_MAX_POSITION)
#     # Choisir une position cible dans la plage
#     if scs_comm_result == COMM_SUCCESS and scs_comm_result2 == COMM_SUCCESS and scs_error == 0 and scs_error2 == 0:
#         target_pos = min_pos + 100  # Un peu au-dessus du min
#         if target_pos < min_pos or target_pos > max_pos:
#             print(f"ATTENTION: La position cible {target_pos} est hors plage [{min_pos}, {max_pos}] !")
#         else:
#             print(f"Envoi de la position {target_pos} (plage [{min_pos}, {max_pos}])")
#         # Activation du torque
#         scs_comm_result3, scs_error3 = packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_SCS_TORQUE_ENABLE, 1)
#         if scs_comm_result3 != COMM_SUCCESS:
#             print("Erreur activation torque:", packetHandler.getTxRxResult(scs_comm_result3))
#         elif scs_error3 != 0:
#             print("Erreur activation torque:", packetHandler.getRxPacketError(scs_error3))
#         else:
#             print("Torque activé")
#         # Envoi de la position
#         scs_comm_result4, scs_error4 = packetHandler.write2ByteTxRx(portHandler, motor_id, ADDR_SCS_GOAL_POSITION, target_pos)
#         if scs_comm_result4 != COMM_SUCCESS:
#             print("Erreur d'envoi:", packetHandler.getTxRxResult(scs_comm_result4))
#         elif scs_error4 != 0:
#             print("Erreur d'envoi:", packetHandler.getRxPacketError(scs_error4))
#         else:
#             print("Commande de position envoyée avec succès.")
#     else:
#         print("Impossible de lire la plage min/max pour ce moteur.")
            
# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()
    
# Enable SCServo torque
scs_comm_result, scs_error = packetHandler.write1ByteTxRx(portHandler, SCS_ID, ADDR_SCS_TORQUE_ENABLE, 1)
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("%s" % packetHandler.getRxPacketError(scs_error))
print("torque enabled")

# Enable SCServo torque
scs_comm_result, scs_error = packetHandler.write1ByteTxRx(portHandler, SCS_ID, ADDR_SCS_TORQUE_ENABLE, 1)
if scs_comm_result != COMM_SUCCESS:
    print("Erreur activation torque:", packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("Erreur activation torque:", packetHandler.getRxPacketError(scs_error))
else:
    print("Torque activé")

# Write SCServo acc
scs_comm_result, scs_error = packetHandler.write1ByteTxRx(portHandler, SCS_ID, ADDR_SCS_GOAL_ACC, SCS_MOVING_ACC)
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("%s" % packetHandler.getRxPacketError(scs_error))

# Write SCServo speed
scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, SCS_ID, ADDR_SCS_GOAL_SPEED, SCS_MOVING_SPEED)
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("%s" % packetHandler.getRxPacketError(scs_error))
print("speed modifiee")
while 1:
    print("Press any key to continue! (or press ESC to quit!)")
    if getch() == chr(0x1b):
        break

    # # Test: envoyer une position à tous les moteurs détectés
    # test_position = 2300
    # print(f"Envoi de la position {test_position} à tous les moteurs détectés...")
    # for motor_id in ids_found:
    #     print(f"\nMoteur ID {motor_id} :")
    #     # Activation du torque
    #     scs_comm_result, scs_error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_SCS_TORQUE_ENABLE, 1)
    #     if scs_comm_result != COMM_SUCCESS:
    #         print("Erreur activation torque:", packetHandler.getTxRxResult(scs_comm_result))
    #     elif scs_error != 0:
    #         print("Erreur activation torque:", packetHandler.getRxPacketError(scs_error))
    #     else:
    #         print("Torque activé")
    #     # Envoi de la position
    #     scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, motor_id, ADDR_SCS_GOAL_POSITION, test_position)
    #     if scs_comm_result != COMM_SUCCESS:
    #         print("Erreur d'envoi:", packetHandler.getTxRxResult(scs_comm_result))
    #     elif scs_error != 0:
    #         print("Erreur d'envoi:", packetHandler.getRxPacketError(scs_error))
    #     else:
    #         print("Commande de position envoyée avec succès.")

    while 1:
        # Read SCServo present position
        scs_present_position_speed, scs_comm_result, scs_error = packetHandler.read4ByteTxRx(portHandler, SCS_ID, ADDR_SCS_PRESENT_POSITION)
        if scs_comm_result != COMM_SUCCESS:
            print(packetHandler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print(packetHandler.getRxPacketError(scs_error))

        scs_present_position = SCS_LOWORD(scs_present_position_speed)
        scs_present_speed = SCS_HIWORD(scs_present_position_speed)
        print("[ID:%03d] GoalPos:%03d PresPos:%03d PresSpd:%03d" 
              % (SCS_ID, scs_goal_position[index], scs_present_position, SCS_TOHOST(scs_present_speed, 15)))

        if not (abs(scs_goal_position[index] - scs_present_position_speed) > SCS_MOVING_STATUS_THRESHOLD):
            break


    # Change goal position
    if index == 0:
        index = 1
    else:
        index = 0    

scs_comm_result, scs_error = packetHandler.write1ByteTxRx(portHandler, SCS_ID, ADDR_SCS_TORQUE_ENABLE, 0)
if scs_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(scs_comm_result))
elif scs_error != 0:
    print("%s" % packetHandler.getRxPacketError(scs_error))
# Close port
portHandler.closePort()