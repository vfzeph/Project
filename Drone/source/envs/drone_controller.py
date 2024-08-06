import airsim
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DroneController:
    def __init__(self, client: airsim.MultirotorClient, logger=None):
        self.client = client
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Drone controller initialized")

    def move_by_velocity_z(self, vx, vy, vz, duration):
        self.client.moveByVelocityZAsync(float(vx), float(vy), float(vz), float(duration), airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, 0)).join()
        time.sleep(duration)
    
    def moveByVelocityZAsync(self, vx, vy, z, duration):
        try:
            self.logger.info(f"Attempting to move: vx={vx}, vy={vy}, z={z}, duration={duration}")
            task = self.client.moveByVelocityZAsync(float(vx), float(vy), float(z), float(duration), airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, 0))
            if task is not None:
                task.join()
                self.logger.info("Move completed successfully")
                return True
            else:
                self.logger.warning("moveByVelocityZAsync returned None")
                return False
        except Exception as e:
            self.logger.error(f"Error in moveByVelocityZAsync: {str(e)}")
            return False

    def rotateByYawRateAsync(self, yaw_rate, duration):
        try:
            self.logger.info(f"Attempting to rotate: yaw_rate={yaw_rate}, duration={duration}")
            task = self.client.rotateByYawRateAsync(float(yaw_rate), float(duration))
            if task is not None:
                task.join()
                self.logger.info("Rotation completed successfully")
                return True
            else:
                self.logger.warning("rotateByYawRateAsync returned None")
                return False
        except Exception as e:
            self.logger.error(f"Error in rotateByYawRateAsync: {str(e)}")
            return False
        
    def move_by_velocity(self, vx, vy, vz, duration):
        logger.info(f"Moving with velocity: vx={vx}, vy={vy}, vz={vz}, duration={duration}")
        self.client.moveByVelocityAsync(float(vx), float(vy), float(vz), float(duration), airsim.DrivetrainType.MaxDegreeOfFreedom).join()
        logger.info("Move complete")

    def log_drone_state(self):
        state = self.getMultirotorState()
        position = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        velocity = state.kinematics_estimated.linear_velocity
        logger.info(f"Drone state - Position: x={position.x_val:.2f}, y={position.y_val:.2f}, z={position.z_val:.2f}")
        logger.info(f"Orientation: x={orientation.x_val:.2f}, y={orientation.y_val:.2f}, z={orientation.z_val:.2f}, w={orientation.w_val:.2f}")
        logger.info(f"Velocity: x={velocity.x_val:.2f}, y={velocity.y_val:.2f}, z={velocity.z_val:.2f}")

    def is_moving(self, threshold=0.1):
        velocity = self.get_velocity()
        speed = np.linalg.norm([velocity.x_val, velocity.y_val, velocity.z_val])
        return speed > threshold

    def getMultirotorState(self):
        return self.client.getMultirotorState()

    def get_collision_info(self):
        return self.client.simGetCollisionInfo()

    def get_position(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def get_velocity(self):
        vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val])

    def get_orientation(self):
        orientation = self.client.getMultirotorState().kinematics_estimated.orientation
        return np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val])

    def get_angular_velocity(self):
        angular_vel = self.client.getMultirotorState().kinematics_estimated.angular_velocity
        return np.array([angular_vel.x_val, angular_vel.y_val, angular_vel.z_val])

    def reset(self):
        logger.info("Resetting drone")
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        logger.info("Drone reset complete")

    def takeoff(self, altitude: float = 3.0):
        logger.info(f"Taking off to altitude: {altitude}m")
        self.client.takeoffAsync(timeout_sec=5).join()
        self.client.moveToZAsync(-altitude, 1).join()
        logger.info("Takeoff complete")

    def land(self):
        logger.info("Landing drone")
        self.client.landAsync(timeout_sec=5).join()
        self.client.armDisarm(False)
        logger.info("Landing complete")

    def move_to_position(self, x: float, y: float, z: float, velocity: float = 5):
        start_pos = self.get_position()
        logger.info(f"Moving from position: x={start_pos[0]:.2f}, y={start_pos[1]:.2f}, z={start_pos[2]:.2f}")
        logger.info(f"Moving to position: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        self.client.moveToPositionAsync(x, y, z, velocity).join()
        end_pos = self.get_position()
        logger.info(f"Move complete. Final position: x={end_pos[0]:.2f}, y={end_pos[1]:.2f}, z={end_pos[2]:.2f}")

    def rotate(self, angle: float):
        logger.info(f"Rotating by {angle} degrees")
        yaw = self.get_orientation()[2] + angle
        self.client.rotateToYawAsync(yaw, timeout_sec=5).join()
        logger.info("Rotation complete")

    def get_image(self):
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        return img_rgb

    def execute_trajectory(self, trajectory: list):
        logger.info("Executing trajectory")
        for point in trajectory:
            self.move_to_position(*point)
            time.sleep(1)  # Short pause between movements
        logger.info("Trajectory execution complete")

    def emergency_stop(self):
        logger.warning("Emergency stop initiated")
        self.client.hoverAsync().join()
        logger.warning("Drone stopped and hovering")

    def __del__(self):
        self.client.enableApiControl(False)
        logger.info("Drone controller deactivated")

    def get_state(self):
        try:
            position = self.drone_controller.get_position()
            velocity = self.drone_controller.get_velocity()
            orientation = self.drone_controller.get_orientation()
            angular_velocity = self.drone_controller.get_angular_velocity()
            collision_info = self.drone_controller.get_collision_info()

            state = np.concatenate([
                position,
                velocity,
                orientation,
                angular_velocity,
                np.array([int(collision_info.has_collided)]),
                np.array([self.current_step]),
                np.array([self.get_gps_data()])
            ])

            assert len(state) == self.state_dim, f"State dimension mismatch. Expected {self.state_dim}, got {len(state)}"
            return state
        except Exception as e:
            self.logger.error(f"Error in get_state: {str(e)}")
            raise