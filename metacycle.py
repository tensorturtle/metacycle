import carla
from carla import ColorConverter as cc

import os
import argparse
import datetime
import logging
import math
import random
import re
import weakref
import numpy as np
import pygame

from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_BACKSPACE
from pygame.locals import K_COMMA
from pygame.locals import K_DOWN
from pygame.locals import K_ESCAPE
from pygame.locals import K_F1
from pygame.locals import K_LEFT
from pygame.locals import K_PERIOD
from pygame.locals import K_RIGHT
from pygame.locals import K_SPACE
from pygame.locals import K_TAB
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_b
from pygame.locals import K_c
from pygame.locals import K_d
from pygame.locals import K_l
from pygame.locals import K_m
from pygame.locals import K_q
from pygame.locals import K_r
from pygame.locals import K_s
from pygame.locals import K_v
from pygame.locals import K_w

pycycling_input = None

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name
 
def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except Exception:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- Bluetooth Cycling Equipment Inputs ----------------------------------------
# ==============================================================================

from dataclasses import dataclass
@dataclass
class EquipmentInputs:
    power: float = 0.0
    cadence: float = 0.0
    heart_rate: float = 0.0
    connected: bool = False

# ==============================================================================
# -- Simulation Outputs --------------------------------------------------------
# ==============================================================================
@dataclass
class SimulationOutputs:
    server_fps: int = 0
    client_fps: int = 0
    map: str = ''
    elapsed_time: datetime.timedelta = datetime.timedelta(0.0)
    speed: float = 0.0
    compass: tuple[float, str] = (0.0, '')
    accel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)
    location: tuple[float, float] = (0.0, 0.0)
    gnss: tuple[float, float] = (0.0, 0.0)
    height: float = 0.0
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    reverse: bool = False
    hand_brake: bool = False
    manual: bool = False
    gear: int = 0
    road_gradient: float = 0.0

import time

class RoadGradientEstimator:
    """
    A class to estimate the road gradient using a moving average filter.
    Includes functionality to ignore calculations for the first 'n' updates
    and to skip frames for more stable gradient calculations.
    """

    def __init__(self, window_size=20, ignore_first_n=0, skip_frames=15):
        """
        Initialize the RoadGradientEstimator with specified parameters.
        Args:
            window_size (int): The number of gradients to keep for averaging.
            ignore_first_n (int): The number of initial updates to ignore.
            skip_frames (int): The number of frames to skip between gradient calculations. At around 60fps, the time between frames is too small and and dividing a very small change in altitude by a small time delta causes numerical instability. Recommended to set it something like 1/4 or 1/2 second.
        """
        self.window_size = window_size
        self.ignore_first_n = ignore_first_n
        self.skip_frames = max(1, skip_frames)  # Ensure skip_frames is at least 1
        self.update_count = 0
        self.frame_count = 0
        self.last_elevation = None
        self.last_time = None
        self.gradients = []
        self.accumulated_distance = 0
        self.accumulated_elevation = 0

    def update(self, elevation, speed):
        """
        Update the estimator with new elevation and speed data.
        Calculates gradient only after skipping the specified number of frames.

        Args:
            elevation (float): Elevation in meters.
            speed (float): Horizontal speed in meters/second.

        Returns:
            float: Average road gradient in percentage, or None if calculation is not possible.
        """
        self.update_count += 1
        self.frame_count += 1
        current_time = time.time()

        if self.last_time is not None:
            delta_time = current_time - self.last_time
            distance = speed * delta_time
            self.accumulated_distance += distance
            self.accumulated_elevation += elevation - self.last_elevation

        if self.frame_count >= self.skip_frames:
            if (self.update_count > self.ignore_first_n and 
                self.last_time is not None and 
                self.accumulated_distance > 0.1):
                current_gradient = (self.accumulated_elevation / self.accumulated_distance) * 100
                self._add_gradient(current_gradient)
            
            # Reset accumulations and frame count
            self.frame_count = 0
            self.accumulated_distance = 0
            self.accumulated_elevation = 0

        self.last_elevation = elevation
        self.last_time = current_time

        return self.get_average_gradient() if self.gradients else 0.0

    def _add_gradient(self, gradient):
        """
        Add a new gradient to the list and maintain the size within the window.
        """
        self.gradients.append(gradient)
        if len(self.gradients) > self.window_size:
            self.gradients.pop(0)

    def get_average_gradient(self):
        """
        Calculate and return the average gradient from the recent data.
        """
        if not self.gradients:
            return 0
        return sum(self.gradients) / len(self.gradients)

# ==============================================================================
# -- Bluetooth -----------------------------------------------------------------
# ==============================================================================
# Add these imports at the top
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import sys
import subprocess

from bleak import BleakScanner, BleakClient
from bleak.exc import BleakDBusError, BleakError
from pycycling.sterzo import Sterzo
from pycycling.fitness_machine_service import FitnessMachineService



class BLECyclingService(Enum):
    '''
    BLE Services and Characteristics that are broadcasted by the devices themselves when being scanned.

    Some are assigned: https://www.bluetooth.com/specifications/assigned-numbers/
    Others (like STERZO) are just made up by the manufacturer.
    '''
    STERZO = "347b0001-7635-408b-8918-8ff3949ce592"
    FITNESS = "00001826-0000-1000-8000-00805f9b34fb"
    POWERMETER = "00001818-0000-1000-8000-00805f9b34fb"

class LiveControlState:
    def __init__(self):
        self.steer = 0
        self.watts = 0
        self.cadence = 0
        self.throttle = 0
        self.brake = 0
        self.wheel_speed = 0

    def update_steer(self, steer):
        '''
        Convert degrees (from BLE) to normalized (-1, 1) (for CARLA)

        STERZO has a range of ~60 degrees
        '''
        self.steer = steer / 150 # experimentally determined.

        self.steer = steer / 150
    def update_throttle(self, watts):
        '''
        Convert watts (from BLE) to normalized (0, 1) (for CARLA)
        '''
        self.watts = watts
        self.throttle = watts / 100 # experimentally determined
        self.throttle = watts / 100

    def update_speed(self, speed):
        '''
        No conversion; speed is km/h throughout.
        '''
        self.wheel_speed = speed

    def update_cadence(self, cadence):
        '''
        No conversion; cadence is RPM throughout.
        '''
        self.cadence = cadence

class PycyclingInput:
    def __init__(self, sterzo_device, powermeter_device, on_steering_update, on_power_update, on_speed_update, on_cadence_update):
        '''
        sterzo_device: BLEDevice
        powermeter_device: BLEDevice
        on_steering_update: callback function (used to send steering angle to carla client)
        on_power_update: callback function (used to send power to carla client)
        '''
        self.sterzo_device = sterzo_device
        self.powermeter_device = powermeter_device
        self.sterzo_device = sterzo_device
        self.powermeter_device = powermeter_device
        self.on_steering_update = on_steering_update
        self.on_power_update = on_power_update
        self.on_speed_update = on_speed_update
        self.on_cadence_update = on_cadence_update

        self.ftms = None
        self.ftms_max_resistance = None
        self.ftms_desired_resistance = 0

    async def run_all(self):
        loop = asyncio.get_running_loop()
        # loop.create_task(self.connect_to_powermeter())
        loop.create_task(self.connect_to_sterzo())
        loop.create_task(self.connect_to_fitness_machine())
        await asyncio.Future()


    async def connect_to_sterzo(self):
        try:
            async with BleakClient(self.sterzo_device) as client:
                def steering_handler(steering_angle):
                    #print(f"Steering angle: {steering_angle}")
                    self.on_steering_update(steering_angle)

                await client.is_connected()
                sterzo = Sterzo(client)
                sterzo.set_steering_measurement_callback(steering_handler)
                await sterzo.enable_steering_measurement_notifications()
                await asyncio.Future()
        except (BleakDBusError, BleakError) as e:
            print(f"Bluetooth connection error (Sterzo): {e}")
            return

    # async def connect_to_powermeter(self):
    #     async with BleakClient(self.powermeter_device) as client:
    #         def power_handler(power):
    #             #print(f"Power: {power.instantaneous_power}")
    #             self.on_power_update(power.instantaneous_power)
    #             self.socketio.emit('power', power.instantaneous_power)
    #             self.socketio.emit('power_device', self.powermeter_device.name)

    #         await client.is_connected()
    #         powermeter = CyclingPowerService(client)
    #         powermeter.set_cycling_power_measurement_handler(power_handler)
    #         await powermeter.enable_cycling_power_measurement_notifications()
    #         await asyncio.sleep(1e10) # run forever

    async def connect_to_fitness_machine(self):
        try:
            async with BleakClient(self.powermeter_device, timeout=20) as client:
                # long timeout is required. Somehow FTMS takes longer to setup.
                await client.is_connected()

                self.ftms = FitnessMachineService(client)
                print("Connected to FTMS")

                res_levels = await self.ftms.get_supported_resistance_level_range()
                print(f"Resistance level range: {res_levels}")
                self.ftms_max_resistance = res_levels.maximum_resistance

                def print_control_point_response(message):
                    pass
                    # print("Received control point response:")
                    # print(message)
                    # print()
                self.ftms.set_control_point_response_handler(print_control_point_response)


                def print_indoor_bike_data(data):
                    # print("Received indoor bike data:")
                    # print(data)
                    power = data.instant_power
                    self.on_power_update(power)

                    speed = data.instant_speed
                    self.on_speed_update(speed)

                    cadence = data.instant_cadence
                    self.on_cadence_update(cadence)

                self.ftms.set_indoor_bike_data_handler(print_indoor_bike_data)
                await self.ftms.enable_indoor_bike_data_notify()

                fitness_machine_features = await self.ftms.get_fitness_machine_feature()

                if not fitness_machine_features.resistance_level_supported:
                    print("WARNING: Resistance level not supported on this smart trainer.")
                    return

                if not fitness_machine_features.resistance_level_supported:
                    print("WARNING: Resistance level not supported on this smart trainer.")
                    return


                await self.ftms.enable_control_point_indicate()
                await self.ftms.request_control()
                await self.ftms.reset()

                while True:
                    if self.ftms_desired_resistance > self.ftms_max_resistance:
                        print("Warning: Desired resistance is greater than max resistance. Setting to max resistance.")
                        self.ftms_desired_resistance = self.ftms_max_resistance
                    #print(f"Setting resistance to {self.ftms_desired_resistance}")
                    await self.ftms.set_target_resistance_level(self.ftms_desired_resistance)
                    await asyncio.sleep(1)
        except (BleakDBusError, BleakError) as e:
            print(f"Bluetooth connection error (FTMS): {e}")
            return

class BluetoothManager:
    def __init__(self, equipment_inputs, live_control_state):
        self.equipment_inputs = equipment_inputs
        self.live_control_state = live_control_state
        self.client = None
        self.device = None
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.loop = asyncio.new_event_loop()

        self.restart_system_bluetooth()

    def restart_system_bluetooth(self):
        '''
        On Ubuntu, after ungraceful shutdown, the bluetooth needs to be restarted from the OS
        '''
        if sys.platform.startswith("linux"):
            print("Restarting system bluetooth")
            subprocess.call(["bluetoothctl", "power", "off"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.call(["bluetoothctl", "power", "on"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        elif sys.platform == 'win32':
            print("Skipping restart of system bluetooth because it is not required on Windows")

        else:
            print("Unknown OS")

    def filter_cycling_accessories(self, devices):
        relevant_devices = {
            'sterzos': [],
            'smart_trainers': [],
        }

        print(f"Number of devices found: {len(devices)}")

        for k,v in devices.items():
            bledevice, advertisement_data = v
            services = advertisement_data.service_uuids

            if BLECyclingService.STERZO.value in services:
                relevant_devices['sterzos'].append(bledevice)
            if BLECyclingService.FITNESS.value in services and BLECyclingService.POWERMETER.value in services:
                relevant_devices['smart_trainers'].append(bledevice)
        print(f"Found {len(relevant_devices['sterzos'])} sterzos and {len(relevant_devices['smart_trainers'])} smart trainers")
        return relevant_devices

    async def scan_and_connect(self):
        global pycycling_input
        while self.running:
            try:
                if not self.equipment_inputs.connected:
                    # Scan for devices
                    devices = await BleakScanner().discover(timeout=1.0, return_adv=True)
                    cycling_ble_devices = self.filter_cycling_accessories(devices)
                    if len(cycling_ble_devices['sterzos']) > 0 and len(cycling_ble_devices['smart_trainers']) > 0:
                        pycycling_input = PycyclingInput(
                            # TODO: What if there are multiple devices for each category?
                            cycling_ble_devices['sterzos'][0],
                            cycling_ble_devices['smart_trainers'][0],
                            on_steering_update=self.live_control_state.update_steer,
                            on_power_update=self.live_control_state.update_throttle,
                            on_speed_update=self.live_control_state.update_speed,
                            on_cadence_update=self.live_control_state.update_cadence
                        )

                        await pycycling_input.run_all()

                await asyncio.sleep(2)
            except Exception as e:
                print(f"Bluetooth error: {e}")
                self.equipment_inputs.connected = False
                await asyncio.sleep(1)

    def start(self):
        """Start the Bluetooth manager in a separate thread"""
        self.executor.submit(self._run_async_loop)

    def _run_async_loop(self):
        """Run the async event loop in the executor thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.scan_and_connect())

    def stop(self):
        """Stop the Bluetooth manager"""
        self.running = False
        self.loop.stop()
        self.executor.shutdown(wait=False)

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, sim_outputs, args):
        self.world = carla_world
        self.sim_outputs = sim_outputs
        self.actor_role_name = "hero"
        self._server_clock = pygame.time.Clock()
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = 'vehicle.diamondback.century'
        self._actor_generation = '2'
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(self.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]
        self.road_gradient_estimator = RoadGradientEstimator(window_size=5, ignore_first_n=30)

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        blueprint = random.choice(blueprint_list)
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        self.world.wait_for_tick()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, clock):
        if self.player is not None:
            t = self.player.get_transform()
            v = self.player.get_velocity()
            c = self.player.get_control()
            compass = self.imu_sensor.compass
            heading = 'N' if compass > 270.5 or compass < 89.5 else ''
            heading += 'S' if 90.5 < compass < 269.5 else ''
            heading += 'E' if 0.5 < compass < 179.5 else ''
            heading += 'W' if 180.5 < compass < 359.5 else ''

            self.sim_outputs.server_fps = self.server_fps
            self.sim_outputs.client_fps = clock.get_fps()
            self.sim_outputs.map = self.map.name.split('/')[-1]
            self.sim_outputs.elapsed_time = datetime.timedelta(seconds=int(self.simulation_time))
            self.sim_outputs.speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            self.sim_outputs.compass = (compass, heading)
            self.sim_outputs.accel = self.imu_sensor.accelerometer
            self.sim_outputs.gyro = self.imu_sensor.gyroscope
            self.sim_outputs.location = (t.location.x, t.location.y)
            self.sim_outputs.gnss = (self.gnss_sensor.lat, self.gnss_sensor.lon)
            self.sim_outputs.height = t.location.z

            # displaying not implemented
            self.sim_outputs.throttle = c.throttle
            self.sim_outputs.steer = c.steer
            self.sim_outputs.brake = c.brake
            self.sim_outputs.reverse = c.reverse
            self.sim_outputs.hand_brake = c.hand_brake
            self.sim_outputs.manual = c.manual_gear_shift
            self.sim_outputs.gear = c.gear

            road_gradient = self.road_gradient_estimator.update(
                t.location.z, # elevation in meters
                math.sqrt(v.x**2 + v.y**2), # keep m/s
            )
            self.sim_outputs.road_gradient = road_gradient if road_gradient is not None else 0.0

        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()

# ==============================================================================
# -- PycyclingControl ----------------------------------------------------------
# ==============================================================================

class PycyclingControl():
    def __init__(self, world, sim_outputs, live_control_state):
        self._control = carla.VehicleControl()
        self.world = world
        self.sim_outputs = sim_outputs
        self.live_control_state = live_control_state

    def step(self):
        self._control.throttle = 0.5

        # == Steering =========================================================
        steer = self.live_control_state.steer
        current_speed = self.sim_outputs.speed
        steering_magic_number = 40
        sensitivity = math.exp(-current_speed/steering_magic_number)
        # Steering sensitivity decreases with: y=e^(-x)
        # for magic_number = 40,
        # at 80km/h, sensitivity is 0.14
        # at 40km/h, sensitivity is 0.36
        # at 20km/h, sensitivity is 0.61
        # at 10km/h, sensitivity is 0.77
        steer *= sensitivity

        self._control.steer = steer


        # == Throttle from Power ===============================================
        # Apply road gradient to wheel_speed
        # Simulate downhill speed gain by increasing wheel speed
        # Uphill speed decrease does not need to be simulated because the
        # physics engine already does this and the smart trainer increases resistance.
        # for each 1% gradient steeper than -3%, add 5km/h to wheel speed (up to 50km/h additional)
        road_gradient = self.sim_outputs.road_gradient

        if road_gradient < -3.0:
            added_speed = (((-1 * road_gradient) - 3.0) * 5)
            if added_speed > 50:
                added_speed = 50
            self.live_control_state.wheel_speed += added_speed


        # use throttle and brake to adjust speed
        # using set_target_velocity() doesn't work because it doesn't interact with the vehicle dynamics
        # despite its inelegance, this works pretty well.
        if self.live_control_state.wheel_speed > current_speed + 3:
            #print("Throttle hard")
            self._control.throttle = 1
            self._control.brake = 0
        elif self.live_control_state.wheel_speed > current_speed + 0.5:
            #print("Throttle soft")
            self._control.throttle = 0.5
            self._control.brake = 0
        elif self.live_control_state.wheel_speed < current_speed + 0.5:
            #print("Coast")
            self._control.throttle = 0
            self._control.brake = 0
        else:
            # actually same as coast because friction is pretty high
            #print("Brake")
            self._control.throttle = 0
            self._control.brake = 0.0

        self.world.player.apply_control(self._control)


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world):
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        self.world = world
        self._steer_cache = 0.0
        self.world.player.set_light_state(self._lights)


    def handle_event(self, event):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if self._is_quit_shortcut(event.key):
                return True
            elif event.key == K_BACKSPACE:
                self.world.restart()
            elif event.key == K_F1:
                self.world.hud.toggle_info()
            elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                self.world.next_map_layer(reverse=True)
            elif event.key == K_v:
                self.world.next_map_layer()
            elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                self.world.load_map_layer(unload=True)
            elif event.key == K_b:
                self.world.load_map_layer()
            elif event.key == K_TAB:
                self.world.camera_manager.toggle_camera()
            elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                self.world.next_weather(reverse=True)
            elif event.key == K_c:
                self.world.next_weather()
            if event.key == K_q:
                self._control.gear = 1 if self._control.reverse else -1
            elif self._control.manual_gear_shift and event.key == K_COMMA:
                self._control.gear = max(-1, self._control.gear - 1)
            elif self._control.manual_gear_shift and event.key == K_PERIOD:
                self._control.gear = self._control.gear + 1
            elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                current_lights ^= carla.VehicleLightState.Special1
            elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                current_lights ^= carla.VehicleLightState.HighBeam
            elif event.key == K_l:
                # Use 'L' key to switch between lights:
                # closed -> position -> low beam -> fog
                if not self._lights & carla.VehicleLightState.Position:
                    self.world.hud.notification("Position lights")
                    current_lights |= carla.VehicleLightState.Position
                else:
                    self.world.hud.notification("Low beam lights")
                    current_lights |= carla.VehicleLightState.LowBeam
                if self._lights & carla.VehicleLightState.LowBeam:
                    self.world.hud.notification("Fog lights")
                    current_lights |= carla.VehicleLightState.Fog
                if self._lights & carla.VehicleLightState.Fog:
                    self.world.hud.notification("Lights off")
                    current_lights ^= carla.VehicleLightState.Position
                    current_lights ^= carla.VehicleLightState.LowBeam
                    current_lights ^= carla.VehicleLightState.Fog

        self._control.reverse = self._control.gear < 0
        # Set automatic control-related vehicle lights
        if self._control.brake:
            current_lights |= carla.VehicleLightState.Brake
        else: # Remove the Brake flag
            current_lights &= ~carla.VehicleLightState.Brake
        if self._control.reverse:
            current_lights |= carla.VehicleLightState.Reverse
        else: # Remove the Reverse flag
            current_lights &= ~carla.VehicleLightState.Reverse
        if current_lights != self._lights: # Change the light state only if necessary
            self._lights = current_lights
            self.world.player.set_light_state(carla.VehicleLightState(self._lights))


    def parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.1, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

                # Apply control
        self.world.player.apply_control(self._control)

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD():
    def __init__(self, width, height, sim_outputs, live_control_state):
        self.sim_outputs = sim_outputs
        self.live_control_state = live_control_state
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []

        self._font_bottom_panel = pygame.font.Font(mono, 36)
        self._bottom_panel_height = 300
        self._bottom_panel_margin_h = 500 # horizontal margin
        self._bottom_panel_margin_v = 70 # vertical margin
        self._large_power_font = pygame.font.Font(mono, 144)


    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return


        self._info_text = [
            f'Server:  {self.sim_outputs.server_fps:16.0f} FPS',
            f'Client:  {self.sim_outputs.client_fps:16.0f} FPS',
            '',
            f'Map:     {self.sim_outputs.map:20s}',
            f'Simulation time: {str(self.sim_outputs.elapsed_time):>12s}',
            '',
            f'Speed:   {self.sim_outputs.speed:15.0f} km/h',
            f'Compass: {self.sim_outputs.compass[0]:17.0f}° {self.sim_outputs.compass[1]:2s}',
            f'Accelero: ({self.sim_outputs.accel[0]:5.1f},{self.sim_outputs.accel[1]:5.1f},{self.sim_outputs.accel[2]:5.1f})',
            f'Gyroscop: ({self.sim_outputs.gyro[0]:5.1f},{self.sim_outputs.gyro[1]:5.1f},{self.sim_outputs.gyro[2]:5.1f})',
            f'Location: ({self.sim_outputs.location[0]:5.1f}, {self.sim_outputs.location[1]:5.1f})',
            f'GNSS: ({self.sim_outputs.gnss[0]:2.6f}, {self.sim_outputs.gnss[1]:3.6f})',
            f'Height:  {self.sim_outputs.height:18.0f} m',
            f'Road Gradient: {self.sim_outputs.road_gradient:.2f}',
            '',
            f'Power: {self.live_control_state.watts}',
            f'Cadence: {self.live_control_state.cadence}',
            f'Wheel speed: {self.live_control_state.wheel_speed}',
            '',
        ]

        self._bottom_panel_text = [
            f'Power: {self.live_control_state.watts}',
            f'Speed: {int(self.sim_outputs.speed):d} km/h',
            f'Gradient: {int(self.sim_outputs.road_gradient):d}%',
            f'RPM: {int(self.live_control_state.cadence):d} RPM',
            f'Time: {str(self.sim_outputs.elapsed_time)}',
        ]

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18

            # Calculate bottom panel dimensions with margins
            panel_width = self.dim[0] - 2 * self._bottom_panel_margin_h
            panel_height = self._bottom_panel_height - 2 * self._bottom_panel_margin_v
            panel_x = self._bottom_panel_margin_h
            panel_y = self.dim[1] - self._bottom_panel_height + self._bottom_panel_margin_v

            # Render bottom panel background
            bottom_panel_surface = pygame.Surface((panel_width, panel_height))
            bottom_panel_surface.set_alpha(100)
            bottom_panel_surface.fill((0, 0, 0))  # Fill with black color
            display.blit(bottom_panel_surface, (panel_x, panel_y))

            # Render power (large) on the left column
            power_surface = self._large_power_font.render(f'{self.live_control_state.watts}W', True, (255, 255, 255))
            power_rect = power_surface.get_rect(center=(panel_x + panel_width // 4, panel_y + panel_height // 2))
            display.blit(power_surface, power_rect)

            # Render other bottom panel text on the right column
            right_column_x = panel_x + panel_width // 2
            v_offset = panel_y + 20  # Start from the top of the panel
            v_spacing = (panel_height - 40) // (len(self._bottom_panel_text) - 1)  # Subtract 1 to exclude power

            for item in self._bottom_panel_text[1:]:  # Skip the power item
                surface = self._font_bottom_panel.render(item, True, (255, 255, 255))
                text_rect = surface.get_rect(left=right_column_x, top=v_offset)
                display.blit(surface, text_rect)
                v_offset += v_spacing


        self._notifications.render(display)



# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)



# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        Attachment = carla.AttachmentType

        self._camera_transforms = [
            # x is 'meters in front of rider'
            # z is 'meters from ground above rider'
            (carla.Transform(carla.Location(x=-3.5, z=2.5), carla.Rotation(pitch=20.0)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=0.1, z=1.2), carla.Rotation(pitch=-92.0, yaw=180.0)), Attachment.SpringArmGhost),

        ]
        self.transform_index = 0
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],

        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- Button -------------------------------------------------------------------
# ==============================================================================

class Button:
    def __init__(self, x, y, width, height, text, font_size=14, callback=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.Font(pygame.font.get_default_font(), font_size)
        self.is_hovered = False
        self.is_pressed = False
        self.callback = callback

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.is_hovered:
                self.is_pressed = True
                if self.callback:
                    self.callback()
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False
        return False

    def draw(self, surface):
        # Button colors
        if self.is_pressed:
            color = (100, 100, 100)  # Darker when pressed
        elif self.is_hovered:
            color = (150, 150, 150)  # Lighter when hovered
        else:
            color = (200, 200, 200)  # Default color

        # Draw button rectangle
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (100, 100, 100), self.rect, 2)  # Border

        # Render text
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class NumericInput:
    def __init__(self, x, y, width, height, min_value, max_value, initial_value, step=1):
        pygame.font.init()

        self.rect = pygame.Rect(x, y, width, height)
        self.min_value = min_value
        self.max_value = max_value
        self.value = initial_value
        self.step = step
        self.font = pygame.font.Font(pygame.font.get_default_font(), 24)
        
        button_width = height
        self.decr_button = pygame.Rect(x, y, button_width, height)
        self.incr_button = pygame.Rect(x + width - button_width, y, button_width, height)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.decr_button.collidepoint(event.pos):
                self.value = max(self.min_value, self.value - self.step)
            elif self.incr_button.collidepoint(event.pos):
                self.value = min(self.max_value, self.value + self.step)
        
    def draw(self, surface):
        # Draw background
        pygame.draw.rect(surface, (200, 200, 200), self.rect)
        
        # Draw decrement button
        pygame.draw.rect(surface, (150, 150, 150), self.decr_button)
        decr_text = self.font.render("-", True, (0, 0, 0))
        surface.blit(decr_text, (self.decr_button.centerx - decr_text.get_width() // 2, 
                                 self.decr_button.centery - decr_text.get_height() // 2))
        
        # Draw increment button
        pygame.draw.rect(surface, (150, 150, 150), self.incr_button)
        incr_text = self.font.render("+", True, (0, 0, 0))
        surface.blit(incr_text, (self.incr_button.centerx - incr_text.get_width() // 2, 
                                 self.incr_button.centery - incr_text.get_height() // 2))
        
        # Draw value
        value_text = self.font.render(f"{self.value:.1f}%", True, (0, 0, 0))
        value_rect = value_text.get_rect(center=self.rect.center)
        surface.blit(value_text, value_rect)\
        
# ==============================================================================
# -- GPX Creator ---------------------------------------------------------------
# ==============================================================================
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re

class GPXCreator:
    def __init__(self, output_dir, creator_name="Metacycle"):
        self.output_dir = output_dir
        self.gpx = ET.Element("gpx", {
            "creator": creator_name,
            "version": "1.1",
            "xmlns": "http://www.topografix.com/GPX/1/1",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": "http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd http://www.garmin.com/xmlschemas/GpxExtensions/v3 http://www.garmin.com/xmlschemas/GpxExtensionsv3.xsd http://www.garmin.com/xmlschemas/TrackPointExtension/v1 http://www.garmin.com/xmlschemas/TrackPointExtensionv1.xsd",
            "xmlns:gpxtpx": "http://www.garmin.com/xmlschemas/TrackPointExtension/v1",
            "xmlns:gpxx": "http://www.garmin.com/xmlschemas/GpxExtensions/v3"
        })

        self.metadata = ET.SubElement(self.gpx, "metadata")
        self.trk = ET.SubElement(self.gpx, "trk")
        self.trkseg = ET.SubElement(self.trk, "trkseg")

    def set_metadata_time(self, time):
        time_element = ET.SubElement(self.metadata, "time")
        time_element.text = time

    def set_track_info(self, name, type):
        name_element = ET.SubElement(self.trk, "name")
        name_element.text = name

        type_element = ET.SubElement(self.trk, "type")
        type_element.text = type

    def add_trackpoint(self, lat, lon, ele, time, power=None, cadence=None):
        trkpt = ET.SubElement(self.trkseg, "trkpt", {"lat": str(lat), "lon": str(lon)})
        
        ele_element = ET.SubElement(trkpt, "ele")
        ele_element.text = str(ele)

        time_element = ET.SubElement(trkpt, "time")
        time_element.text = time

        if power is not None or cadence is not None:
            extensions = ET.SubElement(trkpt, "extensions")

        if power is not None:
            power_element = ET.SubElement(extensions, "power")
            power_element.text = str(power)

        if cadence is not None:
            cadence_element = ET.SubElement(extensions, "cadence")
            cadence_element.text = str(cadence)
        


    def to_string(self):
        rough_string = ET.tostring(self.gpx, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def sanitize_file_name(self, file_name):
        # Replace invalid characters on windows with an underscore
        return re.sub(r'[<>:"/\\|?*]', '_', file_name)

    def save_to_file(self, file_name):
        # Get the user's home directory in a cross-platform way
        home_dir = os.path.expanduser("~")
        
        # Define Downloads path based on OS
        if os.name == "nt":  # Windows
            downloads_dir = os.path.join(home_dir, "Downloads")
        else:  # Linux, macOS, etc.
            downloads_dir = os.path.join(home_dir, "Downloads")  # Most Unix systems use "Downloads"
            if not os.path.exists(downloads_dir):  # Fallback for systems using XDG
                try:
                    import subprocess
                    xdg_dir = subprocess.check_output(["xdg-user-dir", "DOWNLOAD"]).decode().strip()
                    if os.path.exists(xdg_dir):
                        downloads_dir = xdg_dir
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass  # Stick with default Downloads directory if XDG lookup fails
        
        # Create Downloads directory if it doesn't exist
        os.makedirs(downloads_dir, exist_ok=True)
        
        sanitized_name = self.sanitize_file_name(file_name)
        full_path = os.path.normpath(os.path.join(downloads_dir, sanitized_name))
        
        with open(full_path, "w") as file:
            file.write(self.to_string())
        
        print(f"Ride saved to {full_path} - you can upload this to Strava!")
        
        return full_path  # Return the path so the caller knows where the file was saved


# ==============================================================================
# -- list_maps() ---------------------------------------------------------------
# ==============================================================================

def list_maps(carla_client):
    try:
        available_maps = carla_client.get_available_maps()
        # sort maps alphabetically
        available_maps = sorted([m.split('/')[-1] for m in available_maps])
        return available_maps
    except Exception as e:
        print(f"Error getting available maps: {e}")
        return []

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    global pycycling_input

    road_gradient_offset = 0.0
    gradient_input = NumericInput(10, args.height - 300, 200, 40, -15, 15, 0, step=0.5)


    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    bluetooth_manager = None


    try:
        def toggle_control_mode():
            nonlocal keyboard_override
            keyboard_override = not keyboard_override
            mode = "Keyboard" if keyboard_override else "PyCycling"
            hud.notification(f'Switched to {mode} control mode')

        def toggle_camera_view():
            world.camera_manager.toggle_camera()
            hud.notification('Camera View Changed')

        # Create buttons
        button_width = 200
        button_height = 40
        button_margin = 10
        button_x = 10
        extra_bottom_margin = 70
        button_y_start = args.height - (2 * button_height + button_margin + 10 + extra_bottom_margin)

        toggle_control_button = Button(button_x, button_y_start, button_width, button_height, "Toggle Control Mode", callback=toggle_control_mode)
        toggle_camera_button = Button(button_x, button_y_start + button_height + button_margin, button_width, button_height, "Toggle Camera View", callback=toggle_camera_view)

            
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        sim_world = client.get_world()

        # client.load_world('Town07')

        # Get maps and respect user's input args
        default_map = "Town07"
        map = args.map if args.map in list_maps(client) else default_map
        client.load_world(map)

        gpx_creator = GPXCreator("finished_gpx")
        gpx_creator.set_metadata_time(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
        gpx_creator.set_track_info("Virtual Cycling Activity in Metacycle", "VirtualRide")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SCALED | pygame.RESIZABLE)
        display.fill((0,0,0))
        pygame.display.flip()

        sim_outputs = SimulationOutputs()
        equipment_inputs = EquipmentInputs()  # Add this lineBluetoothManager
        # pycycling_input = None # is later instantiated with the right BLE Devices by BluetoothManager
        # Initialize and start Bluetooth manager
        live_control_state = LiveControlState()
        bluetooth_manager = BluetoothManager(equipment_inputs, live_control_state)
        bluetooth_manager.start()

        hud = HUD(args.width, args.height, sim_outputs, live_control_state)
        world = World(sim_world, hud, sim_outputs, args)
        controller = KeyboardControl(world)
        pycycling_controller = PycyclingControl(world, sim_outputs, live_control_state)

        # Create toggle button
        toggle_button = Button(10, args.height - 100, 200, 40, "Toggle Control Mode")

        sim_world.wait_for_tick()

        clock = pygame.time.Clock()

        keyboard_override = False

        gpx_frame_counter = 0
        while True:
            clock.tick_busy_loop(args.max_fps)
            pressed_keys = pygame.key.get_pressed()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                toggle_control_button.handle_event(event)
                toggle_camera_button.handle_event(event)
                if controller.handle_event(event):
                    return
                gradient_input.handle_event(event)

            # vehicle movement key presses
            if keyboard_override:
                controller.parse_vehicle_keys(pressed_keys, clock.get_time())
            else:
                pycycling_controller.step()

            # This should be modified by user (not implemented)
            road_gradient_offset = gradient_input.value
            # send resistance
            if pycycling_input is not None:
                # resistance is 0-200
                # max resistance is reached at 15 percent road gradient
                # resistance is linearly interpolated between 0 and 15 percent road gradient
                # cutoff at 15 percent road gradient
                # if negative road gradient, resistance is 0
                gradient = sim_outputs.road_gradient + road_gradient_offset
                gradient = max(0, gradient)
                gradient = min(15, gradient)

                pycycling_input.ftms_desired_resistance = (gradient * 200 / 15) # 200 is maximum resistance, 15 is maximum gradient
                pycycling_input.ftms_desired_resistance += (sim_outputs.speed * 1.3) # wind resistance estimate based on speed. Magic number empirically determined. Should be exposed to user as a preference.


            world.tick(clock)
            world.render(display)

            # Draw the buttons
            toggle_control_button.draw(display)
            toggle_camera_button.draw(display)

            # Draw the numeric input
            gradient_input.draw(display)
            input_text = "Gradient Offset:"
            text_surface = pygame.font.Font(None, 24).render(input_text, True, (255, 255, 255))
            display.blit(text_surface, (10, args.height - 320))


            pygame.display.flip()

            gpx_frame_counter += 1
            if gpx_frame_counter % 60 == 0: # don't create a GPX point for every frame, it messes up strava
                # Match it to be basically 1Hz
                # North-south offsets change the distance travelled, so stick to places close to the equator.
                gnss_offset = (-0.849541, -91.104870) # Galapagos islands

                gpx_creator.add_trackpoint(
                    sim_outputs.gnss[0] + gnss_offset[0],
                    sim_outputs.gnss[1] + gnss_offset[1],
                    sim_outputs.height,
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}.{int(time.time() * 1000 % 1000)}Z", # we need millisecond precision because there are multiple trackpoints per second. Using seconds causes speed calculation problems in Strava.
                    live_control_state.watts or None, # uses OR short-circuiting
                    live_control_state.cadence or None,
                )
                gpx_frame_counter = 0

    finally:
        gpx_creator.save_to_file(f"metacycle_ride_{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}.gpx")

        if bluetooth_manager:
            bluetooth_manager.stop()

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1920x1080',
        help='window resolution (default: 1920x1080)')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--max-fps',
        default=60,
        type=int,
        help="Frame rate of monitor. Defines the max speed of (default: 60)"
    )
    argparser.add_argument(
        '--map', '-m',
        default="Town07",
        help="Map selection. Choose from available CARLA maps."
    )
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        print(args)

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
