"""
Robot: binds a RobotSpec to a simulation scene and owns the robot's sensors,
actuators and state snapshot.

Lifecycle (mirrors engine build semantics):
    robot = Robot(spec, scene, device)     # adds articulation to the scene
    scene.build(n_envs)
    robot.bind(n_envs)                     # resolve DOFs, create sensors/actuators
    ...
    robot.refresh()                        # once per control step, then read robot.state
"""

from __future__ import annotations

import torch

from domo.sim.base import Scene

from .actuators import PDJointPositionActuator
from .sensors import SimBaseStateSensor, SimContactSensor, SimIMU, SimJointEncoders
from .spec import RobotSpec
from .state import RobotState

__all__ = ["Robot"]


class Robot:
    def __init__(self, spec: RobotSpec, scene: Scene, device: torch.device,
                 kp: float | None = None, kd: float | None = None,
                 base_init_pos: tuple | None = None,
                 base_init_quat: tuple | None = None):
        self.spec = spec
        self.device = device
        self._kp = kp if kp is not None else spec.kp
        self._kd = kd if kd is not None else spec.kd

        init_pos = base_init_pos if base_init_pos is not None else spec.base_init_pos
        init_quat = base_init_quat if base_init_quat is not None else spec.base_init_quat
        self.base_init_pos = torch.tensor(init_pos, device=device, dtype=torch.float32)
        self.base_init_quat = torch.tensor(init_quat, device=device,
                                           dtype=torch.float32)

        self.articulation = scene.add_articulation(
            spec.urdf_path, tuple(init_pos), tuple(init_quat))

        # Populated by bind():
        self.n_envs = 0
        self.dof_idx = None
        self.default_dof_pos = None
        self.state: RobotState | None = None
        self.actuator: PDJointPositionActuator | None = None
        self._state_sensors = []

    # ------------------------------------------------------------------

    def bind(self, n_envs: int) -> None:
        """Resolve structure and allocate state. Call after scene.build()."""
        spec = self.spec
        self.n_envs = n_envs
        self.dof_idx = self.articulation.dof_indices(spec.joint_names)
        self.default_dof_pos = torch.tensor(
            spec.default_dof_angles, device=self.device, dtype=torch.float32)

        self.actuator = PDJointPositionActuator(
            self.articulation, self.dof_idx, self._kp, self._kd)

        n_feet = len(spec.foot_link_names)
        self.state = RobotState.zeros(n_envs, spec.num_dofs, max(n_feet, 1), self.device)

        imu = SimIMU(self.articulation, self.device)
        encoders = SimJointEncoders(self.articulation, self.dof_idx)
        base_state = SimBaseStateSensor(self.articulation)
        # Order matters: IMU first (others use state.base_quat).
        self._state_sensors = [imu, encoders, base_state]

        # Foot links may not exist (URDF importers often merge fixed links)
        # and the backend may not expose contact forces — both optional.
        self.contact_sensor = None
        if n_feet > 0:
            try:
                foot_idx = self.articulation.link_indices(spec.foot_link_names)
                contact = SimContactSensor(self.articulation, foot_idx)
            except Exception:
                contact = None
            if contact is not None and contact.available:
                self.contact_sensor = contact
                self._state_sensors.append(contact)

    # ------------------------------------------------------------------

    def refresh(self) -> RobotState:
        """Pull fresh measurements from all state sensors into self.state."""
        for sensor in self._state_sensors:
            sensor.update(self.state)
        return self.state

    def set_joint_targets(self, targets: torch.Tensor) -> None:
        self.actuator.apply(targets)

    # ------------------------------------------------------------------

    def reset_idx(self, envs_idx: torch.Tensor,
                  base_pos: torch.Tensor | None = None) -> None:
        """
        Reset selected envs to the default configuration.
        base_pos: optional [len(envs_idx), 3] spawn positions (e.g. terrain
        grid); defaults to the spec's init position.
        """
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        state = self.state

        state.dof_pos[envs_idx] = self.default_dof_pos
        state.dof_vel[envs_idx] = 0.0
        self.articulation.set_joint_positions(
            state.dof_pos[envs_idx], self.dof_idx, envs_idx, zero_velocity=True)

        if base_pos is None:
            base_pos = self.base_init_pos.unsqueeze(0).expand(n, -1)
        state.base_pos[envs_idx] = base_pos
        state.base_quat[envs_idx] = self.base_init_quat
        self.articulation.set_base_pose(
            state.base_pos[envs_idx], state.base_quat[envs_idx], envs_idx)

        state.base_lin_vel[envs_idx] = 0.0
        state.base_lin_vel_world[envs_idx] = 0.0
        state.base_ang_vel[envs_idx] = 0.0
        self.articulation.zero_all_velocities(envs_idx)
