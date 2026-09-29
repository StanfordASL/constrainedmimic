"""Utilities for interfacing with MuJoCo"""

from typing import Union

import mujoco
import mujoco.viewer
import numpy as np


def visualize_transform(
    viewer: mujoco.viewer.Handle,
    transform: np.ndarray,
    scale: float = 0.2,
    width: float = 0.01,
    alpha: float = 1.0,
    reset_ngeom: bool = False,
) -> None:
    """Visualize a transformation matrix (or batch of transforms) in the MuJoCo viewer

    Args:
        viewer (mujoco.viewer.Handle): MuJoCo viewer handle, launched in passive mode
        transform (np.ndarray): Transformation matrix to visualize, shape (4, 4).
            Can also pass in a batch of transforms to view, shape (N, 4, 4)
        scale (float, optional): Scale of the frame(s). Defaults to 0.2.
        width (float, optional): Width of the arrows in the frame. Defaults to 0.01.
        alpha (float, optional): Transparency. Defaults to 1.0
        reset_ngeom (bool, optional): Whether to reset the ngeoms flag in the viewer
            prior to visualization (can help manage resources in the viewer). Defaults
            to False.
    """
    transform = np.asarray(transform)
    return visualize_frame(
        viewer,
        transform[..., :3, 3],
        transform[..., :3, :3],
        scale,
        width,
        alpha,
        reset_ngeom,
    )


def visualize_frame(
    viewer: mujoco.viewer.Handle,
    pos: np.ndarray,
    rmat: np.ndarray,
    scale: float = 0.2,
    width: float = 0.01,
    alpha: float = 1.0,
    reset_ngeom: bool = False,
) -> None:
    """Visualize a frame (or batch of frames) in the MuJoCo viewer

    Args:
        viewer (mujoco.viewer.Handle): MuJoCo viewer handle, launched in passive mode
        pos (np.ndarray): Position(s), shape (..., 3)
        rmat (np.ndarray): Rotation matrix / matrices, shape (..., 3, 3)
        scale (float, optional): Scale of the frame(s). Defaults to 0.2.
        width (float, optional): Width of the arrows in the frame. Defaults to 0.01.
        alpha (float, optional): Transparency. Defaults to 1.0
        reset_ngeom (bool, optional): Whether to reset the ngeoms flag in the viewer
            prior to visualization (can help manage resources in the viewer). Defaults
            to False.
    """
    if reset_ngeom:
        viewer.user_scn.ngeom = 0
    pos = np.asarray(pos)
    rmat = np.asarray(rmat)
    assert pos.ndim in [1, 2]
    assert rmat.ndim in [2, 3]
    assert pos.shape[-1] == 3
    assert rmat.shape[-2:] == (3, 3)
    is_batched = pos.ndim == 2 or rmat.ndim == 3
    if is_batched:
        N = pos.shape[0]
        assert rmat.shape[0] == N
        for i in range(N):
            _visualize_frame_single(viewer, pos[i], rmat[i], scale, width, alpha)
    else:
        _visualize_frame_single(viewer, pos, rmat, scale, width, alpha)


def _visualize_frame_single(
    viewer: mujoco.viewer.Handle,
    pos: np.ndarray,
    rmat: np.ndarray,
    scale: float = 0.2,
    width: float = 0.01,
    alpha: float = 1.0,
) -> None:
    """Helper function to visualize a single frame"""
    rgbas = [[1, 0, 0, alpha], [0, 1, 0, alpha], [0, 0, 1, alpha]]
    for i in range(3):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            print("Scene geom buffer full!")
            break
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[width, width, scale],
            pos=pos,
            mat=rmat.flatten(),
            rgba=rgbas[i],
        )
        endpoint = pos + rmat[:, i] * scale
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            width,
            pos,
            endpoint,
        )
        viewer.user_scn.ngeom += 1


def _visualize_box_single(
    viewer: mujoco.viewer.Handle,
    pos: np.ndarray,
    rmat: np.ndarray,
    size: np.ndarray,
    rgba: np.ndarray,
) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        print("Scene geom buffer full!")
        return
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
        mat=rmat.flatten(),
        rgba=rgba,
    )
    viewer.user_scn.ngeom += 1


def visualize_box(
    viewer: mujoco.viewer.Handle,
    pos: np.ndarray,
    rmat: np.ndarray,
    size: np.ndarray,
    rgba: np.ndarray,
    reset_ngeom: bool = False,
):
    if reset_ngeom:
        viewer.user_scn.ngeom = 0
    size = np.asarray(size)
    pos = np.asarray(pos)
    rmat = np.asarray(rmat)
    assert size.shape[-1] == 3
    assert pos.shape[-1] == 3
    assert rmat.shape[-2:] == (3, 3)
    is_batched = pos.ndim == 2 or rmat.ndim == 3
    if is_batched:
        N = pos.shape[0]
        assert rmat.shape[0] == N
        assert size.shape[0] == N
        assert rgba.shape[0] == N
        for i in range(N):
            _visualize_box_single(viewer, pos[i], rmat[i], size[i], rgba[i])
    else:
        _visualize_box_single(viewer, pos, rmat, size, rgba)


def _visualize_sphere_single(
    viewer: mujoco.viewer.Handle,
    pos: np.ndarray,
    size: float,
    rgba: np.ndarray,
) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        print("Scene geom buffer full!")
        return
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, 0, 0]),
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    viewer.user_scn.ngeom += 1


def visualize_sphere(
    viewer: mujoco.viewer.Handle,
    pos: np.ndarray,
    size: Union[float, np.ndarray],
    rgba: np.ndarray,
    reset_ngeom: bool = False,
):
    if reset_ngeom:
        viewer.user_scn.ngeom = 0
    size = np.asarray(size)
    pos = np.asarray(pos)
    assert pos.shape[-1] == 3
    is_batched = pos.ndim == 2
    if is_batched:
        N = pos.shape[0]
        assert size.shape[0] == N
        assert rgba.shape[0] == N
        for i in range(N):
            _visualize_sphere_single(viewer, pos[i], size[i], rgba[i])
    else:
        _visualize_sphere_single(viewer, pos, size, rgba)


def _visualize_cylinder_single(
    viewer: mujoco.viewer.Handle,
    pos: np.ndarray,
    rmat: np.ndarray,
    radius: float,
    size: float,  # half the length of the cylinder
    rgba: np.ndarray,
) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        print("Scene geom buffer full!")
        return
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([radius, size, 0]),
        pos=pos,
        mat=rmat.flatten(),
        rgba=rgba,
    )
    viewer.user_scn.ngeom += 1


def visualize_cylinder(
    viewer: mujoco.viewer.Handle,
    pos: np.ndarray,
    rmat: np.ndarray,
    radius: Union[float, np.ndarray],
    size: Union[float, np.ndarray],
    rgba: np.ndarray,
    reset_ngeom: bool = False,
):
    if reset_ngeom:
        viewer.user_scn.ngeom = 0
    size = np.asarray(size)
    pos = np.asarray(pos)
    rmat = np.asarray(rmat)
    assert pos.shape[-1] == 3
    assert rmat.shape[-2:] == (3, 3)
    is_batched = pos.ndim == 2 or rmat.ndim == 3
    if is_batched:
        N = pos.shape[0]
        assert rmat.shape[0] == N
        assert radius.shape[0] == N
        assert size.shape[0] == N
        assert rgba.shape[0] == N
        for i in range(N):
            _visualize_cylinder_single(
                viewer, pos[i], rmat[i], radius[i], size[i], rgba[i]
            )
    else:
        _visualize_cylinder_single(viewer, pos, rmat, radius, size, rgba)
