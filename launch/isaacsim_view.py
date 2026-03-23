"""
Minimal Isaac Sim viewer for WujiHand with sin-wave joint targets.

Usage:
    python launch/isaacsim_view.py --side right
    python launch/isaacsim_view.py --side left
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--side", choices=["left", "right"], default="right")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

PKG_DIR = Path(__file__).resolve().parent.parent
SIDE = args_cli.side
HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(PKG_DIR / "usd" / SIDE / "wujihand.usd"),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={f"{SIDE}_finger.*_joint1": 0.06},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[f"{SIDE}_finger.*_joint.*"],
            stiffness=None,  # read from USD
            damping=None,  # read from USD
        ),
    },
)


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 100, device=args_cli.device)
    )
    sim.set_camera_view([0.3, 0.0, 0.3], [0.0, 0.0, 0.05])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2000.0)
    )
    prim_utils.create_prim("/World/hand", "Xform")
    hand = Articulation(cfg=HAND_CFG.replace(prim_path="/World/hand/WujiHand"))

    sim.reset()
    dt = sim.get_physics_dt()
    t = 0.0
    PI2 = 6.28318530

    lo = hand.data.soft_joint_pos_limits[..., 0]
    hi = hand.data.soft_joint_pos_limits[..., 1]
    mid, amp = (lo + hi) / 2, (hi - lo) * 0.3

    phase = torch.zeros(len(hand.joint_names), device=lo.device)
    for i, name in enumerate(hand.joint_names):
        for fi in range(1, 6):
            if f"finger{fi}_" in name:
                phase[i] = (fi - 1) * PI2 / 5
                break

    while simulation_app.is_running():
        target = mid + amp * torch.sin(PI2 * 0.3 * t + phase).unsqueeze(0)
        hand.set_joint_position_target(target.clamp(lo, hi))
        hand.write_data_to_sim()
        sim.step()
        t += dt
        hand.update(dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
